# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
GOSTSASH pressure solvation model.

The GOSTSASH (Gaussian On Surface Tesserae Simulate Anisotropic Strain using Half-shells)
model applies an anisotropic strain to a molecular cavity surface sliced into half-shells using
Gaussian-weighted integrals.

Supports two cavity types:
  - 'vdw': van der Waals surface with scaled Bondi radii
  - 'vdw/occ': Occluded van der Waals surface (crevice-free)

"""

import numpy as np
from functools import cached_property

from pyscf import gto, lib, scf
from pyscf.lib import logger
from pyscf.solvent import _attach_solvent
from pyscf.solvent.pcm import modified_Bondi
from pyscf.solvent.grad.pcm import get_dF_dA
from pyscf.solvent.gostshyp import gen_pressure_surface, MIN_SURFACE_AREA
import os

# Pressure conversion: 1 MPa = 3.3989309735473356e-08 Hartree/Bohr^3
MPA_TO_AU = 3.3989309735473356e-08

def fakemol_for_gaussian(coords, exponents, l=0, cart=True, coeffs=None):
    """Build a fake Mole object representing auxiliary Gaussians.

    Parameters
    ----------
    coords : ndarray of shape (n, 3)
        Gaussian center coordinates in Bohr.
    exponents : ndarray of shape (n,)
        Gaussian exponents.
    l : int
        Angular momentum (0=s, 1=p, 2=d, 3=f).
    cart : bool
        Use Cartesian Gaussians.
    coeffs : ndarray of shape (n,), optional
        Contraction coefficients (default: ones).

    Returns
    -------
    fakemol : gto.Mole
    """
    nbas = coords.shape[0]
    if coeffs is None:
        coeffs = np.ones_like(exponents)
    angmom = np.full(nbas, l, dtype=np.int32)

    ang_norm = {0: 2.0 * np.sqrt(np.pi),
                1: 2.0 * np.sqrt(np.pi / 3),
                2: 1.0,
                3: 1.0}

    fakeatm = np.zeros((nbas, gto.mole.ATM_SLOTS), dtype=np.int32)
    fakebas = np.zeros((nbas, gto.mole.BAS_SLOTS), dtype=np.int32)
    fakeenv = [0] * gto.mole.PTR_ENV_START
    ptr = gto.mole.PTR_ENV_START

    fakeatm[:, gto.mole.PTR_COORD] = np.arange(ptr, ptr + nbas * 3, 3)
    fakeenv.append(coords.ravel())
    ptr += nbas * 3

    fakebas[:, gto.mole.ATOM_OF] = np.arange(nbas)
    fakebas[:, gto.mole.ANG_OF] = angmom
    fakebas[:, gto.mole.NPRIM_OF] = 1
    fakebas[:, gto.mole.NCTR_OF] = 1
    fakebas[:, gto.mole.PTR_EXP] = ptr + np.arange(nbas) * 2
    fakebas[:, gto.mole.PTR_COEFF] = ptr + np.arange(nbas) * 2 + 1

    coeff = ang_norm[l] * coeffs
    fakeenv.append(np.vstack((exponents, coeff)).T.ravel())

    fakemol = gto.Mole()
    fakemol.cart = cart
    fakemol._atm = fakeatm
    fakemol._bas = fakebas
    fakemol._env = np.hstack(fakeenv)
    fakemol._built = True
    return fakemol


def compute_surface_normals(atom_coords, grid, atom_idx):
    """Compute inward-pointing unit normals for surface grid points.

    Parameters
    ----------
    atom_coords : ndarray of shape (natm, 3)
    grid : ndarray of shape (ngrids, 3)
    atom_idx : ndarray of shape (ngrids,)

    Returns
    -------
    normals : ndarray of shape (ngrids, 3)
    """
    ref_coords = atom_coords[atom_idx]
    dr = ref_coords - grid
    dr_norm = np.linalg.norm(dr, axis=1, keepdims=True)
    if np.any(dr_norm < 1e-14):
        raise ValueError(
            'Grid point coincides with its atom center; '
            'surface tessellation is degenerate.')
    return dr / dr_norm


def _is_index(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def parse_plane_groups(atom_list, natm):
    """Validate atom_list and return the three plane points as tuples of atom indices.

    Each of the three entries is an atom index or a list/tuple of atom indices;
    the plane point is then the mean position of these atoms, e.g.
    atom_list = [[9, 10], 4, 2] uses the midpoint of atoms 9 and 10.
    """
    if atom_list is None:
        raise ValueError("You have not specified a list of three atoms (or atom groups) to construct a plane from!")
    if not isinstance(atom_list, (list, tuple)) or len(atom_list) != 3:
        raise ValueError(f"atom_list must be a list of exactly 3 entries (atom index or list of "
                         f"atom indices), got '{atom_list}'")
    groups = []
    for entry in atom_list:
        group = [entry] if _is_index(entry) else entry
        if not isinstance(group, (list, tuple)) or len(group) == 0 or not all(_is_index(i) for i in group):
            raise ValueError(f"Each entry of atom_list must be an atom index or a non-empty list of "
                             f"atom indices of type int, got '{entry}' in '{atom_list}'")
        if len(set(group)) != len(group):
            raise ValueError(f"An atom group must not contain the same atom twice, got '{entry}'")
        for i in group:
            if not 0 <= i < natm:
                raise ValueError(f"The atom index {i} does not exist in this molecule!")
        groups.append(tuple(int(i) for i in group))
    if len({frozenset(g) for g in groups}) != 3:
        raise ValueError(f"The three plane points must be built from different atoms (groups), got: {atom_list}!")
    return tuple(groups)


def plane_point_weights(groups, natm):
    """Weights W (3, natm) with plane points a_k = W[k] @ atom_coords: W[k, B] = 1/|group_k|
    if atom B belongs to group k. Also d a_k / d X_B = W[k, B] * identity."""
    W = np.zeros((3, natm))
    for k, g in enumerate(groups):
        W[k, list(g)] = 1.0 / len(g)
    return W


class GOSTSASH(lib.StreamObject):
    """
    GOSTSASH pressure solvation model.

    Attributes
    ----------
    mol : pyscf.gto.Mole
        Molecular object.
    pressure_mpa : float
        Applied pressure in MPa (default: 50000 = 50 GPa).
    npoints : int
        Number of Lebedev grid points per atom (default: 110).
    scaling_factor : float
        Van der Waals radii scaling factor (default: 1.2).
    cavity : str
        Cavity type: 'vdw' or 'vdw/occ' (default: 'vdw/occ').
    r_ext : float
        Extension radius for vdW/OCC in Bohr (default: 0.4724 ~ 0.25 Ang).
    direct : bool
        If True, compute integrals on-the-fly without caching (default: True).
    atom_list : list
        Three points defining the plane (required). Each entry is an atom index
        or a list of atom indices whose mean position is used, e.g.
        [[9, 10], 4, 2] for the midpoint of atoms 9 and 10 (useful for molecules
        without a symmetric CH2 bridge, or ring centres).
    f_exponent : float
        Exponent f of the anisotropic pressure distribution (required, no default).
        Each surface point t gets p_t = P * |d_t|**f / M, where d_t is the distance
        to the plane through the atoms in atom_list and M = sum(A_t |d_t|**f) / sum(A_t)
        is the area-weighted mean, so the area-weighted mean pressure is exactly P.
        Larger f concentrates the load on the poles far from the plane; p_t = P at
        |d| = M**(1/f). Must be >= 1 (the analytic gradient contains |d|**(f-1)).
        There is no plane width: it would cancel out of p_t.
    """

    def __init__(self, mol, options=None):
        self.mol = mol
        self.stdout = mol.stdout
        self.verbose = mol.verbose
        self.max_memory = mol.max_memory

        if options is None:
            options = {}
        self.pressure_mpa = options.get('pressure_mpa', 50_000)
        self.npoints = options.get('npoints', 110)
        self.scaling_factor = options.get('scaling_factor', 1.2)
        self.cavity = options.get('cavity', 'vdw/occ')
        self.r_ext = options.get('r_ext', 0.4724)  # Bohr (0.25 Ang)
        self.direct = options.get('direct', True)
        self.write_grid = options.get('write_grid', False)
        self.output = options.get('output', " ")
        self.atom_list = options.get('atom_list', None)
        # plane points: atom index or mean position of a group of atoms
        self.plane_groups = parse_plane_groups(self.atom_list, mol.natm)
        self.plane_weights = plane_point_weights(self.plane_groups, mol.natm)
        if 'plane_width' in options:
            raise ValueError("The option 'plane_width' was removed: it cancels out of the normalised "
                             "pressure distribution p_t = P |d_t|^f / <|d|^f>_A. Only f_exponent "
                             "controls the anisotropy.")
        # no default on purpose: f has not been benchmarked yet
        self.f_exponent = options.get('f_exponent', None)
        if self.f_exponent is None:
            raise ValueError("GOSTSASH requires the option 'f_exponent' (exponent of the anisotropic "
                             "pressure distribution, e.g. 'f_exponent': 4). There is no default value.")
        if isinstance(self.f_exponent, bool) or not isinstance(self.f_exponent, (int, float)):
            raise ValueError(f"f_exponent must be a number, got '{self.f_exponent}'")
        if self.f_exponent < 1.0:
            raise ValueError(f"f_exponent must be >= 1, got {self.f_exponent}")
        self.f_exponent = float(self.f_exponent)

        self.opt_counter = 0
        self.scf_counter = 0

        self.frozen = False
        self.equilibrium_solvation = False
        self.e = None
        self.v = None
        self.amplitudes = None

        self.build()

    @property
    def pressure_au(self):
        """Pressure in atomic units (Hartree/Bohr^3)."""
        return self.pressure_mpa * MPA_TO_AU

    def build(self, mol=None):
        """Build surface tessellation."""
        print("Building surface")
        if mol is not None:
            self.mol = mol
        mol = self.mol

        rad = self.scaling_factor * modified_Bondi

        if self.cavity == 'vdw/occ':
            r_ext = self.r_ext
            rad_outer = rad + r_ext
            # GOSTSASH always removes tiny-area points (GOSTSHYP: optional 'min_area')
            self.surface_dict, self.n_removed_points = gen_pressure_surface(
                mol, self.npoints, rad_outer, r_ext=r_ext, min_area=MIN_SURFACE_AREA)

            # Save outer surface_dict for gradient computation
            self._outer_surface_dict = dict(self.surface_dict)

            norm_vec = self.surface_dict['norm_vec']
            grid_outer = self.surface_dict['grid_coords']
            grid_inner = grid_outer - r_ext * norm_vec

            R_outer = self.surface_dict['R_vdw']
            R_inner = R_outer - r_ext
            ratio_sq = (R_inner / R_outer) ** 2
            self._occ_ratio_sq = ratio_sq
            area_occ = self.surface_dict['area'] * ratio_sq

            self.surface_dict['grid_coords_outer'] = grid_outer
            self.surface_dict['grid_coords'] = grid_inner
            self.surface_dict['area'] = area_occ
            self.surface_dict['R_vdw'] = R_inner
        else:
            self.surface_dict, self.n_removed_points = gen_pressure_surface(
                mol, self.npoints, rad, min_area=MIN_SURFACE_AREA)
            self._outer_surface_dict = None
            self._occ_ratio_sq = None
        
        self.slice_tessellation_field()

        # Build atom index
        atom_idx = np.zeros(len(self.surface_dict['area']), dtype=np.int32)
        for i, (start, stop) in enumerate(self.surface_dict['gslice_by_atom']):
            atom_idx[start:stop] = i

        self.grid_coords = self.surface_dict['grid_coords']
        if self.write_grid:
            np.savetxt(self.output + f"grid_opt_cycle_{self.opt_counter}.txt", self.grid_coords)
            self.mol.tofile(self.output + f"mol_opt_cycle_{self.opt_counter}.xyz", format = "xyz")
        
        self.areas = self.surface_dict['area']
        if np.any(self.areas <= 0):
            raise ValueError(
                'Non-positive surface areas detected; cavity is degenerate.')
        self.atom_idx = atom_idx
        self.widths = np.pi * np.log(2) / self.areas
        self.n_gaussian = len(self.areas)
        self.surface_normals = compute_surface_normals(
            mol.atom_coords(), self.grid_coords, self.atom_idx)

        # Clear cached properties
        self.__dict__.pop('gtilde', None)
        self.__dict__.pop('force_operators', None)

        logger.info(self, 'GOSTSASH: %d surface Gaussians (cavity=%s)',
                    self.n_gaussian, self.cavity)
        if self.n_removed_points:
            logger.info(self, 'GOSTSASH: %d surface points with area <= %.1e Bohr^2 removed',
                        self.n_removed_points, MIN_SURFACE_AREA)
        return self
    
    def slice_tessellation_field(self):
        # plane points (Bohr): single atoms or mean positions of atom groups
        self.a1, self.a2, self.a3 = self.plane_weights @ self.mol.atom_coords()
        o = self.create_plane()

        # calculate distances of the grid coordinates to the plane
        self.distances = np.dot(self.surface_dict['grid_coords'], self.n0) - o
        self.abs_distances = np.abs(self.distances)

        # p_t = P * |d_t|^f / M, M = sum(A |d|^f) / sum(A): monotonic in the distance to
        # the plane, area-weighted mean exactly P (total load P * sum(A) as in GOSTSHYP).
        # Area weighting keeps the energy continuous when a surface point appears or
        # disappears (its weight goes to zero with its area). A plane width would cancel
        # out of this ratio, so there is none: the pressure equals P at the distance
        # d* = M^(1/f), which follows from f and the geometry.
        area = self.surface_dict['area']
        self.d_pow_f = self.abs_distances ** self.f_exponent
        self.mean_d_pow_f = np.dot(area, self.d_pow_f) / np.sum(area)
        self.pressure_au_list = self.pressure_au * self.d_pow_f / self.mean_d_pow_f

        self.crossover_distance = self.mean_d_pow_f ** (1.0 / self.f_exponent)
        relieved = self.abs_distances < self.crossover_distance
        logger.info(self, f"f_exponent = {self.f_exponent}")
        logger.info(self, f"<|d|^f>_A = {self.mean_d_pow_f:.6e} Bohr^f")
        logger.info(self, f"p_t = P_inp at d* = {self.crossover_distance:.3f} Bohr from the plane "
                          f"(max |d| = {self.abs_distances.max():.3f} Bohr)")
        logger.info(self, f"p_t < P_inp on {100 * area[relieved].sum() / area.sum():.1f} percent of the surface area")
        logger.info(self, f"p_t / P_inp: min = {np.min(self.pressure_au_list) / self.pressure_au:.3e}, "
                          f"max = {np.max(self.pressure_au_list) / self.pressure_au:.3f}")
        logger.info(self, f"P_inp = {self.pressure_au:.14f} a.u.")
        logger.info(self, f"P_eff = {np.dot(area, self.pressure_au_list) / np.sum(area):.14f} a.u. (area-weighted mean)")

    def create_plane(self):
        """
        Creates a plane from 3 points in space.
        p3
        ^ 
        |(v2)
        p1 -(v1)-> p2
        Calculates normal vector and distance d of those 3 points.

        Parameters
        ----------
        p1, p2, p3 : np.array
        """
    
        # the two connecting vectors form the plane
        v1 = self.a2 - self.a1
        v2 = self.a3 - self.a1 

        # cross product of v1 and v2 is orthogonal to them and the normal vector of the plane
        n = np.cross(v1, v2)
        #print(f"normal = {normal}")

        # Normalisation of the normal vector for exact euclidean distances
        self.n_norm = np.linalg.norm(n)
        if np.abs(self.n_norm) < 1e-5:
            raise ValueError("The three atoms lay on a straight line, therefore they cannot form a plane!")
        # normal_normalised directs from the origin in direction of the plane
        self.n0 = n / self.n_norm
 
        # insert point 1 into plane equation: d = n_0 · p1
        # d is distance of plane to origin
        o = np.dot(self.n0, self.a1)

        return o

    def compute_dn0_dX(self):
        """
        Computes the analytic derivative of the normalised plane normal vector n0 with respect to
        all cartesian coordinates. The plane points a_k are atoms or mean positions of atom
        groups, a_k = sum_B W[k, B] X_B, so dn0/dX_B = sum_k W[k, B] dn0/da_k.
            dn0_dX: shape (natm, 3, 3)
        """
        dn0_da = np.zeros((3, 3, 3))   # [plane point k, cartesian c, component of n0]
        for c in range(3):
            ec = np.zeros(3)
            ec[c] = 1.0

            dn_da2 = np.cross(ec, self.a3 - self.a1)
            dn_da3 = np.cross(self.a2 - self.a1, ec)
            dn_da1 = -dn_da2 - dn_da3

            for k, dn in enumerate((dn_da1, dn_da2, dn_da3)):
                dn0_da[k, c] = (dn - np.dot(self.n0, dn) * self.n0) / self.n_norm

        return np.einsum('kB,kcn->Bcn', self.plane_weights, dn0_da)


    def compute_dpressure_dX(self, dareas):
        """
        Derivative of the per-point pressure p_t = P * |d_t|^f / M with the
        area-weighted mean M = sum_t A_t |d_t|^f / sum_t A_t,
        d_t = n0 . (r_t - x1), with respect to all nuclear coordinates
        lambda = (atom, cartesian component):

            dp_t/dlambda = (P / M) * [u_t - |d_t|^f * (dM/dlambda) / M],
            u_t = f * sign(d_t) * |d_t|^(f-1) * dd_t/dlambda,
            dM/dlambda = [sum_t (dA_t/dlambda |d_t|^f + A_t u_t) - M sum_t dA_t/dlambda] / sum_t A_t

        f is a fixed input parameter, so there is no df/dlambda term.
            dareas: dA_t/dlambda of the areas used in the energy, shape (natm, 3, ngrid)
            returns: shape (natm, 3, ngrid)
        """
        mol = self.mol
        grid = self.surface_dict['grid_coords']
        ngrid = len(grid)
        dn0_dX = self.compute_dn0_dX()

        # dd_t/dlambda: grid points move rigidly with their atom; the plane moves with the
        # atoms of its three points (anchor x1 = a1 with weights W[0, i] = 1/|group 1|)
        ddt_dlambda_all = np.zeros((mol.natm, 3, ngrid))
        for i in range(mol.natm):
            for c in range(3):
                dut_dlambda_n0 = np.zeros(ngrid)
                dut_dlambda_n0[self.atom_idx == i] = self.n0[c]

                dx1_dlambda_n0 = self.plane_weights[0, i] * self.n0[c]

                dn0_dlambda_u_t_x1 = np.dot(grid - self.a1, dn0_dX[i, c, :])

                ddt_dlambda_all[i, c, :] = (dut_dlambda_n0 - dx1_dlambda_n0) + dn0_dlambda_u_t_x1

        f = self.f_exponent
        # d(|d_t|^f)/dd_t; f >= 1, so finite also for points on the plane (sign(0) = 0)
        ddf_ddt = f * np.sign(self.distances) * self.abs_distances ** (f - 1)
        u = ddt_dlambda_all * ddf_ddt[np.newaxis, np.newaxis, :]   # d(|d_t|^f)/dlambda

        M = self.mean_d_pow_f
        area = self.surface_dict['area']
        dM_dlambda = (np.einsum('acg,g->ac', dareas, self.d_pow_f)
                      + np.einsum('g,acg->ac', area, u)
                      - M * np.sum(dareas, axis=2)) / np.sum(area)
        return (self.pressure_au / M) * (
            u - self.d_pow_f[np.newaxis, np.newaxis, :] * (dM_dlambda / M)[:, :, np.newaxis])

    def dump_flags(self, verbose=None):
        logger.info(self, '******** %s ********', self.__class__)
        logger.info(self, 'pressure = %.1f MPa (%.6e a.u.)',
                    self.pressure_mpa, self.pressure_au)
        logger.info(self, 'npoints = %d', self.npoints)
        logger.info(self, 'scaling_factor = %.2f', self.scaling_factor)
        logger.info(self, 'cavity = %s', self.cavity)
        if self.cavity == 'vdw/occ':
            logger.info(self, 'r_ext = %.4f Bohr', self.r_ext)
        logger.info(self, 'direct = %s', self.direct)
        logger.info(self, 'n_gaussian = %d', self.n_gaussian)
        logger.info(self, f'atom_list = {self.atom_list}')
        for k, g in enumerate(self.plane_groups, 1):
            logger.info(self, 'plane point %d = %s', k,
                        f'atom {g[0]}' if len(g) == 1 else f'mean of atoms {list(g)}')
        logger.info(self, 'f_exponent = %.4f', self.f_exponent)
        return self

    def check_sanity(self):
        if self.pressure_mpa < 0.:
            raise ValueError(f'pressure_mpa must be positive, got {self.pressure_mpa}')
        if self.scaling_factor < 0.:
            raise ValueError(f'scaling_factor must be positive, got {self.scaling_factor}')
        if self.cavity not in ('vdw', 'vdw/occ'):
            raise ValueError(f"cavity must be 'vdw' or 'vdw/occ', got '{self.cavity}'")
        if self.cavity == 'vdw/occ' and self.r_ext < 0.0:
            raise ValueError(f'r_ext must be positive for vdw/occ cavity, got {self.r_ext}')
        if self.write_grid == True and self.output == " ":
            raise ValueError(f'If grid and amplitudes should be written to file, you have to give a file path for the output')
        if self.write_grid == True and not os.path.isdir(self.output):
            raise ValueError(f"The output file path '{self.output}' you have given does not exist!")
        # atom_list is validated in __init__ (parse_plane_groups)

        return self

    def kernel(self, dm):
        """Compute GOSTSASH energy and Fock matrix contribution.

        Parameters
        ----------
        dm : ndarray of shape (nao, nao) or (2, nao, nao)
            Density matrix.

        Returns
        -------
        energy : float
        fock : ndarray of shape (nao, nao)
        """
        print("Starting SCF cycle")
        if not (isinstance(dm, np.ndarray) and dm.ndim == 2):
            dm = dm[0] + dm[1]

        if self.direct:
            return self._kernel_direct(dm)
        else:
            return self._kernel_cached(dm)

    def _kernel_cached(self, dm):
        """Cached mode: uses precomputed gtilde and force_operators."""
        nao = self.mol.nao_nr()
        dm_flat = dm.ravel(order='F')
        forces = dm_flat @ self.force_operators.reshape(nao * nao, -1, order='F')
        amplitudes = self.pressure_au_list * self.areas / forces
        
        mask = amplitudes < 0
        if np.any(mask):
            n_neg = np.count_nonzero(mask)
            logger.warn(self, 'GOSTSASH: %d/%d (%.1f%%) negative amplitudes removed',
                        n_neg, amplitudes.size, 100.0 * n_neg / amplitudes.size)
            amplitudes[mask] = 0.0
            forces[mask] = 1.0  # avoid division by zero in gradient

        #if np.any(self.slice_mask):
        #    n_sliced = np.count_nonzero(self.slice_mask)
        #    logger.info(self, 'GOSTSASH: %d/%d (%.1f%%) amplitudes have been set to 0 due to slicing',
        #                n_sliced, amplitudes.size, 100.0 * n_sliced / amplitudes.size)
        #    amplitudes[self.slice_mask] = 0.0
        #    forces[self.slice_mask] = 1.0

        self.amplitudes = amplitudes
        
        if self.write_grid:
            np.savetxt(self.output + f"amplitudes_opt_cycle_{self.opt_counter}_scf_cycle_{self.scf_counter}.txt", self.amplitudes)

        self.forces = forces

        gtilde_expval = dm_flat @ self.gtilde.reshape(nao * nao, -1, order='F')
        self.gtilde_expval = gtilde_expval

        fock1 = (self.gtilde.reshape(nao * nao, -1, order='F') @ amplitudes
                 ).reshape(nao, nao, order='F')
        fock2 = -self.pressure_au_list * self.areas * gtilde_expval / (forces ** 2)
        fock2 = (self.force_operators.reshape(nao * nao, -1, order='F') @ fock2
                 ).reshape(nao, nao, order='F')

        energy = np.vdot(fock1, dm)
        fock = fock1 + fock2

        self.e = energy
        self.v = fock
        logger.info(self, 'GOSTSASH energy: %.10f', energy)
        self.scf_counter += 1
        return energy, fock

    def _kernel_direct(self, dm):
        """Integral-direct mode: compute integrals on-the-fly in chunks."""
        mol = self.mol
        nao = mol.nao_nr()

        max_memreq = 5 * self.n_gaussian * nao**2 * 8.0 / 1e6
        max_memory = max(2000, mol.max_memory * 0.9 - lib.current_memory()[0])
        n_chunks = max(1, int(max_memreq // max_memory + 1))

        shells = np.arange(self.n_gaussian)
        chunks = np.array_split(shells, n_chunks)

        gmol = fakemol_for_gaussian(self.grid_coords, self.widths)
        gmol_p = fakemol_for_gaussian(
            self.grid_coords, self.widths, l=1, coeffs=2.0 * self.widths)
        supermol = mol + gmol
        supermol_p = mol + gmol_p

        energy = 0.0
        fock = np.zeros_like(dm)
        self.amplitudes = np.zeros(self.n_gaussian)
        self.forces = np.zeros(self.n_gaussian)
        self.gtilde_expval = np.zeros(self.n_gaussian)

        dm_flat = dm.ravel(order='F')
        nao2 = nao * nao

        for shell_slice in chunks:
            off1 = int(shell_slice[0])
            off2 = len(shell_slice)
            slices = (0, mol.nbas, 0, mol.nbas,
                      mol.nbas + off1, mol.nbas + off1 + off2)

            overlap3_s = supermol.intor('int3c1e', shls_slice=slices, aosym='s1')
            overlap3_p = supermol_p.intor(
                'int3c1e', shls_slice=slices, aosym='s1'
            ).reshape(nao, nao, -1, 3)

            normals = self.surface_normals[shell_slice]
            force_ops = np.einsum('bkgc,gc->bkg', overlap3_p, normals,
                                  optimize=False)
            force_ops_2d = force_ops.reshape(nao2, -1, order='F')
            forces = dm_flat @ force_ops_2d
            amplitudes = self.pressure_au_list[shell_slice] * self.areas[shell_slice] / forces
            
            mask = amplitudes < 0
            if np.any(mask):
                n_neg = np.count_nonzero(mask)
                logger.warn(self, 'GOSTSASH: %d/%d (%.1f%%) negative amplitudes removed',
                            n_neg, amplitudes.size, 100.0 * n_neg / amplitudes.size)
                amplitudes[mask] = 0.0
                forces[mask] = 1.0  # avoid division by zero in gradient

            #if np.any(self.slice_mask[shell_slice]):
            #    n_sliced = np.count_nonzero(self.slice_mask[shell_slice])
            #    logger.info(self, 'GOSTSASH: %d/%d (%.1f%%) amplitudes have been set to 0 due to slicing',
            #            n_sliced, amplitudes.size, 100.0 * n_sliced / amplitudes.size)
            #    amplitudes[self.slice_mask[shell_slice]] = 0.0
            #    forces[self.slice_mask[shell_slice]] = 1.0

            self.amplitudes[shell_slice] = amplitudes
            self.forces[shell_slice] = forces

            overlap3_s_2d = overlap3_s.reshape(nao2, -1, order='F')
            f1 = (overlap3_s_2d @ amplitudes).reshape(nao, nao, order='F')

            gtilde_expval = dm_flat @ overlap3_s_2d
            self.gtilde_expval[shell_slice] = gtilde_expval
            f2 = -self.pressure_au_list[shell_slice] * self.areas[shell_slice] * gtilde_expval / (forces ** 2)
            f2 = (force_ops_2d @ f2).reshape(nao, nao, order='F')

            fock += f1 + f2
            energy += np.vdot(f1, dm)

        if self.write_grid:
            np.savetxt(self.output + f"amplitudes_opt_cycle_{self.opt_counter}_scf_cycle_{self.scf_counter}.txt", self.amplitudes)

        self.e = energy
        self.v = fock
        logger.info(self, 'GOSTSASH energy: %.10f', energy)
        self.scf_counter += 1
        return energy, fock

    @cached_property
    def gtilde(self):
        """s-type 3-center overlap integrals [nao, nao, n_gaussian]."""
        mol = self.mol
        gmol = fakemol_for_gaussian(self.grid_coords, self.widths)
        supermol = mol + gmol
        slices = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas + gmol.nbas)
        ret = supermol.intor('int3c1e', shls_slice=slices, aosym='s1')
        return ret

    @cached_property
    def force_operators(self):
        """p-type 3-center overlaps contracted with normals [nao, nao, n_gaussian]."""
        mol = self.mol
        nao = mol.nao_nr()
        gmol_p = fakemol_for_gaussian(
            self.grid_coords, self.widths, l=1, coeffs=2.0 * self.widths)
        supermol = mol + gmol_p
        slices = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas + gmol_p.nbas)
        overlap3 = supermol.intor(
            'int3c1e', shls_slice=slices, aosym='s1'
        ).reshape(nao, nao, -1, 3)
        force_ops = np.einsum('bkgc,gc->bkg', overlap3, self.surface_normals,
                              optimize=False)
        return force_ops

    def grad(self, dm):
        """Compute analytical nuclear gradient of GOSTSASH energy.

        Parameters
        ----------
        dm : ndarray of shape (nao, nao) or (2, nao, nao)
            Density matrix.

        Returns
        -------
        grad : ndarray of shape (natm, 3)
        """
        print("Calculating gradient")
        self.opt_counter += 1
        if self.forces is None:
            raise RuntimeError(
                'kernel() must be called before grad(). '
                'Forces have not been computed.')

        if not (isinstance(dm, np.ndarray) and dm.ndim == 2):
            dm = dm[0] + dm[1]

        if self.direct:
            return self._grad_direct(dm)
        else:
            return self._grad_cached(dm)

    def _grad_cached(self, dm):
        """Cached gradient: uses precomputed gtilde and force_operators."""
        mol = self.mol

        if self._outer_surface_dict is not None:
            _, dareas = get_dF_dA(self._outer_surface_dict)
            dareas = dareas.transpose(1, 2, 0) * self._occ_ratio_sq
        else:
            _, dareas = get_dF_dA(self.surface_dict)
            dareas = dareas.transpose(1, 2, 0)  # (natm, 3, ngrids)

        forces = self.forces
        gtilde_expval = self.gtilde_expval

        # Term 1: area derivative
        dE1 = np.einsum(
            'g,acg,g->ac', self.pressure_au_list, dareas, gtilde_expval / forces, optimize=True)

        # Term 2: gtilde operator derivative
        gmol = fakemol_for_gaussian(self.grid_coords, self.widths)
        supermol = mol + gmol
        slices = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas + gmol.nbas)

        dPQ = supermol.intor('int3c1e_ip1', shls_slice=slices)
        dPQ = np.einsum('xijn,n->xij', dPQ, self.amplitudes, optimize=True)

        slices_g = (mol.nbas, mol.nbas + gmol.nbas, 0, mol.nbas, 0, mol.nbas)
        dG = supermol.intor('int3c1e_ip1', shls_slice=slices_g)

        aoslice = mol.aoslice_by_atom()
        nao = mol.nao_nr()
        dgtilde_braket = np.einsum('xij,ij->ix', dPQ, dm, optimize=True)
        dgtilde_braket += np.einsum('xij,ji->ix', dPQ, dm, optimize=True)

        dgtilde_gaussian = np.einsum(
            'xnij,n,ij->nx', dG, self.amplitudes, dm, optimize=True)
        gtilde_operator_grad = np.asarray(
            [np.sum(dgtilde_braket[p0:p1], axis=0) for p0, p1 in aoslice[:, 2:]])
        np.add.at(gtilde_operator_grad, self.atom_idx, dgtilde_gaussian)
        gtilde_operator_grad *= -1.0

        # d-orbital width gradient
        wgrad_prefs = -np.pi * np.log(2) / (self.areas ** 2)
        gmol_d = fakemol_for_gaussian(
            self.grid_coords, self.widths, l=2,
            coeffs=wgrad_prefs * self.amplitudes)
        supermol_d = mol + gmol_d
        supermol_d.cart = True
        slices_d = (0, mol.nbas, 0, mol.nbas,
                    mol.nbas, mol.nbas + gmol_d.nbas)

        nao_cart = mol.nao_nr(cart=True)
        overlap3d = supermol_d.intor(
            'int3c1e', shls_slice=slices_d
        ).reshape(nao_cart, nao_cart, -1, 6)

        if not mol.cart:
            c2s = mol.cart2sph_coeff(normalized='sp')
            overlap3d = np.einsum(
                'ij,jkgd,kl->ilgd', c2s.T, overlap3d, c2s, optimize=True)

        diagd = overlap3d[:, :, :, 0] + overlap3d[:, :, :, 3] + overlap3d[:, :, :, 5]
        imd = np.einsum('ijg,ij->g', diagd, dm, optimize=True)
        dE_d = -np.einsum('acg,g->ac', dareas, imd, optimize=True)
        dE2 = gtilde_operator_grad + dE_d

        # Term 3: force operator derivative
        coeffs_fop = (
            -2.0 * self.pressure_au_list * self.areas * gtilde_expval
            * self.widths / (forces * forces))
        gmol_f = fakemol_for_gaussian(
            self.grid_coords, self.widths, l=1, coeffs=coeffs_fop)
        supermol_f = mol + gmol_f
        slices_f = (0, mol.nbas, 0, mol.nbas,
                    mol.nbas, mol.nbas + gmol_f.nbas)
        dpq = supermol_f.intor(
            'int3c1e_ip1', shls_slice=slices_f
        ).reshape(3, nao, nao, -1, 3)
        dpq *= self.surface_normals

        slices_fg = (mol.nbas, mol.nbas + gmol_f.nbas, 0, mol.nbas, 0, mol.nbas)
        dG_f = supermol_f.intor(
            'int3c1e_ip1', shls_slice=slices_fg
        ).reshape(3, -1, 3, nao, nao)

        dpq_ix = np.einsum('xijnp,ij->ix', dpq, dm, optimize=True)
        dpq_ix += np.einsum('xijnp,ji->ix', dpq, dm, optimize=True)

        dG_f = np.einsum('xnpij,np->xnij', dG_f, self.surface_normals, optimize=True)
        dG_f = np.einsum('xnij,ij->nx', dG_f, dm, optimize=True)

        force_operator_grad = np.asarray(
            [np.sum(dpq_ix[p0:p1], axis=0) for p0, p1 in aoslice[:, 2:]])
        np.add.at(force_operator_grad, self.atom_idx, dG_f)
        force_operator_grad *= -1.0

        # f-orbital width gradient for force operators
        coeffs_f2 = -2.0 * self.widths * wgrad_prefs
        gmol_ft = fakemol_for_gaussian(
            self.grid_coords, self.widths, l=3, coeffs=coeffs_f2)
        supermol_ft = mol + gmol_ft
        supermol_ft.cart = True
        slices_ft = (0, mol.nbas, 0, mol.nbas,
                     mol.nbas, mol.nbas + gmol_ft.nbas)
        overlap3f = supermol_ft.intor(
            'int3c1e', shls_slice=slices_ft
        ).reshape(nao_cart, nao_cart, -1, 10)

        if not mol.cart:
            c2s = mol.cart2sph_coeff(normalized='sp')
            overlap3f = np.einsum(
                'ij,jkgd,kl->ilgd', c2s.T, overlap3f, c2s, optimize=True)

        # Trace over Cartesian components -> x, y, z contributions
        xf = overlap3f[:, :, :, 0] + overlap3f[:, :, :, 3] + overlap3f[:, :, :, 5]
        yf = overlap3f[:, :, :, 1] + overlap3f[:, :, :, 6] + overlap3f[:, :, :, 8]
        zf = overlap3f[:, :, :, 2] + overlap3f[:, :, :, 7] + overlap3f[:, :, :, 9]
        dx = np.einsum('ijg,ij->g', xf, dm, optimize=True)
        dy = np.einsum('ijg,ij->g', yf, dm, optimize=True)
        dz = np.einsum('ijg,ij->g', zf, dm, optimize=True)
        dr = np.vstack((dx, dy, dz)).T

        rf2 = 1.0 / (forces * forces)
        dr *= self.surface_normals
        dFdR = dareas * wgrad_prefs * forces / self.widths
        dFdR += np.einsum('gc,axg->axg', dr, dareas, optimize=True)
        width_grad_ftype = (
            -np.einsum(
                'g,g,g,axg,g->ax', self.pressure_au_list, self.areas, gtilde_expval,
                dFdR, rf2, optimize=True))

        dE3 = force_operator_grad + width_grad_ftype

        # Term 4: derivative of the anisotropic pressure distribution
        total_dP_eff = self.compute_dpressure_dX(dareas)

        dE4 = np.einsum(
                    'acg,g,g->ac', total_dP_eff, self.areas, gtilde_expval / forces, optimize=True)

        return dE1 + dE2 + dE3 + dE4

    def _grad_direct(self, dm):
        """Integral-direct gradient: compute integrals on-the-fly in chunks."""
        mol = self.mol
        nao = mol.nao_nr()
        nao_cart = mol.nao_nr(cart=True)
        natm = mol.natm
        aoslice = mol.aoslice_by_atom()

        if self._outer_surface_dict is not None:
            _, dareas = get_dF_dA(self._outer_surface_dict)
            dareas = dareas.transpose(1, 2, 0) * self._occ_ratio_sq
        else:
            _, dareas = get_dF_dA(self.surface_dict)
            dareas = dareas.transpose(1, 2, 0)  # (natm, 3, ngrids)

        forces = self.forces
        gtilde_expval = self.gtilde_expval
        amplitudes = self.amplitudes
        wgrad_prefs = -np.pi * np.log(2) / (self.areas ** 2)

        # Term 1: area derivative (no chunking needed)
        dE1 = np.einsum(
            'acg,g,g->ac', dareas, self.pressure_au_list, gtilde_expval / forces, optimize=True)

        # Chunk size: peak memory is 18 * nao^2 * C * 8 bytes
        max_memory = max(2000, mol.max_memory * 0.9 - lib.current_memory()[0])
        mem_per_grid = 18 * nao**2 * 8.0 / 1e6  # MB per grid point
        chunk_size = max(1, int(max_memory / mem_per_grid))

        if not mol.cart:
            c2s = mol.cart2sph_coeff(normalized='sp')

        shells = np.arange(self.n_gaussian)
        chunks = [shells[i:i+chunk_size]
                  for i in range(0, self.n_gaussian, chunk_size)]

        gtilde_operator_grad = np.zeros((natm, 3))
        dE_d_total = np.zeros((natm, 3))
        force_operator_grad = np.zeros((natm, 3))
        width_grad_ftype = np.zeros((natm, 3))

        for chunk_idx in chunks:
            C = len(chunk_idx)
            g0 = int(chunk_idx[0])

            # --- Build chunk-local fakemols ---
            coords_c = self.grid_coords[chunk_idx]
            widths_c = self.widths[chunk_idx]
            areas_c = self.areas[chunk_idx]
            pressure_au_list_c = self.pressure_au_list[chunk_idx]
            normals_c = self.surface_normals[chunk_idx]
            amplitudes_c = amplitudes[chunk_idx]
            forces_c = forces[chunk_idx]
            gtilde_expval_c = gtilde_expval[chunk_idx]
            wgrad_prefs_c = wgrad_prefs[chunk_idx]
            atom_idx_c = self.atom_idx[chunk_idx]
            dareas_c = dareas[:, :, chunk_idx]

            # --- Term 2: gtilde operator derivative ---
            # s-type ip1 bra/ket
            gmol_s = fakemol_for_gaussian(coords_c, widths_c)
            supermol_s = mol + gmol_s
            slices_s = (0, mol.nbas, 0, mol.nbas,
                        mol.nbas, mol.nbas + gmol_s.nbas)
            slices_sg = (mol.nbas, mol.nbas + gmol_s.nbas,
                         0, mol.nbas, 0, mol.nbas)

            dPQ = supermol_s.intor('int3c1e_ip1', shls_slice=slices_s)
            dPQ = np.einsum('xijn,n->xij', dPQ, amplitudes_c, optimize=True)

            dgtilde_braket = np.einsum('xij,ij->ix', dPQ, dm, optimize=True)
            dgtilde_braket += np.einsum('xij,ji->ix', dPQ, dm, optimize=True)
            del dPQ

            gt_grad_chunk = np.asarray(
                [np.sum(dgtilde_braket[p0:p1], axis=0)
                 for p0, p1 in aoslice[:, 2:]])

            # s-type ip1 Gaussian center
            dG = supermol_s.intor('int3c1e_ip1', shls_slice=slices_sg)
            dgtilde_gaussian = np.einsum(
                'xnij,n,ij->nx', dG, amplitudes_c, dm, optimize=True)
            del dG

            np.add.at(gt_grad_chunk, atom_idx_c, dgtilde_gaussian)
            gt_grad_chunk *= -1.0
            gtilde_operator_grad += gt_grad_chunk
            del dgtilde_braket, gt_grad_chunk, dgtilde_gaussian

            # d-type width gradient
            gmol_d = fakemol_for_gaussian(
                coords_c, widths_c, l=2,
                coeffs=wgrad_prefs_c * amplitudes_c)
            supermol_d = mol + gmol_d
            supermol_d.cart = True
            slices_d = (0, mol.nbas, 0, mol.nbas,
                        mol.nbas, mol.nbas + gmol_d.nbas)
            overlap3d = supermol_d.intor(
                'int3c1e', shls_slice=slices_d
            ).reshape(nao_cart, nao_cart, C, 6)

            if not mol.cart:
                overlap3d = np.einsum(
                    'ij,jkgd,kl->ilgd', c2s.T, overlap3d, c2s, optimize=True)

            diagd = (overlap3d[:, :, :, 0] + overlap3d[:, :, :, 3]
                     + overlap3d[:, :, :, 5])
            imd = np.einsum('ijg,ij->g', diagd, dm, optimize=True)
            dE_d_total -= np.einsum('acg,g->ac', dareas_c, imd, optimize=True)
            del overlap3d, diagd, imd

            # --- Term 3: force operator derivative ---
            coeffs_fop_c = (
                -2.0 * pressure_au_list_c * areas_c * gtilde_expval_c
                * widths_c / (forces_c * forces_c))

            # p-type ip1 bra/ket
            gmol_f = fakemol_for_gaussian(
                coords_c, widths_c, l=1, coeffs=coeffs_fop_c)
            supermol_f = mol + gmol_f
            slices_f = (0, mol.nbas, 0, mol.nbas,
                        mol.nbas, mol.nbas + gmol_f.nbas)
            dpq = supermol_f.intor(
                'int3c1e_ip1', shls_slice=slices_f
            ).reshape(3, nao, nao, C, 3)
            dpq *= normals_c

            dpq_ix = np.einsum('xijnp,ij->ix', dpq, dm, optimize=True)
            dpq_ix += np.einsum('xijnp,ji->ix', dpq, dm, optimize=True)
            del dpq

            fop_grad_chunk = np.asarray(
                [np.sum(dpq_ix[p0:p1], axis=0)
                 for p0, p1 in aoslice[:, 2:]])

            # p-type ip1 Gaussian center
            slices_fg = (mol.nbas, mol.nbas + gmol_f.nbas,
                         0, mol.nbas, 0, mol.nbas)
            dG_f = supermol_f.intor(
                'int3c1e_ip1', shls_slice=slices_fg
            ).reshape(3, C, 3, nao, nao)

            dG_f = np.einsum(
                'xnpij,np->xnij', dG_f, normals_c, optimize=True)
            dG_f = np.einsum('xnij,ij->nx', dG_f, dm, optimize=True)

            np.add.at(fop_grad_chunk, atom_idx_c, dG_f)
            fop_grad_chunk *= -1.0
            force_operator_grad += fop_grad_chunk
            del dpq_ix, fop_grad_chunk, dG_f

            # f-type width gradient for force operators
            coeffs_f2_c = -2.0 * widths_c * wgrad_prefs_c
            gmol_ft = fakemol_for_gaussian(
                coords_c, widths_c, l=3, coeffs=coeffs_f2_c)
            supermol_ft = mol + gmol_ft
            supermol_ft.cart = True
            slices_ft = (0, mol.nbas, 0, mol.nbas,
                         mol.nbas, mol.nbas + gmol_ft.nbas)
            overlap3f = supermol_ft.intor(
                'int3c1e', shls_slice=slices_ft
            ).reshape(nao_cart, nao_cart, C, 10)

            if not mol.cart:
                overlap3f = np.einsum(
                    'ij,jkgd,kl->ilgd', c2s.T, overlap3f, c2s, optimize=True)

            xf = (overlap3f[:, :, :, 0] + overlap3f[:, :, :, 3]
                  + overlap3f[:, :, :, 5])
            yf = (overlap3f[:, :, :, 1] + overlap3f[:, :, :, 6]
                  + overlap3f[:, :, :, 8])
            zf = (overlap3f[:, :, :, 2] + overlap3f[:, :, :, 7]
                  + overlap3f[:, :, :, 9])
            dx = np.einsum('ijg,ij->g', xf, dm, optimize=True)
            dy = np.einsum('ijg,ij->g', yf, dm, optimize=True)
            dz = np.einsum('ijg,ij->g', zf, dm, optimize=True)
            dr = np.vstack((dx, dy, dz)).T
            del overlap3f, xf, yf, zf

            rf2_c = 1.0 / (forces_c * forces_c)
            dr *= normals_c
            dFdR = dareas_c * wgrad_prefs_c * forces_c  / widths_c
            dFdR += np.einsum('gc,axg->axg', dr, dareas_c, optimize=True)
            width_grad_ftype -= np.einsum(
                'g,g,g,axg,g->ax', pressure_au_list_c, areas_c, gtilde_expval_c,
                dFdR, rf2_c, optimize=True)

        dE2 = gtilde_operator_grad + dE_d_total
        dE3 = force_operator_grad + width_grad_ftype

        # Term 4: derivative of the anisotropic pressure distribution
        total_dP_eff = self.compute_dpressure_dX(dareas)

        dE4 = np.einsum(
                    'acg,g,g->ac', total_dP_eff, self.areas, gtilde_expval / forces, optimize=True)

        return dE1 + dE2 + dE3 + dE4

    def reset(self, mol=None):
        """Reset for geometry optimization / scanner."""
        print("Reset for geometry optimization or scanner")
        if mol is not None:
            self.mol = mol
        self.__dict__.pop('gtilde', None)
        self.__dict__.pop('force_operators', None)
        self.e = None
        self.v = None
        self.amplitudes = None
        self.forces = None
        self.build()
        self.scf_counter = 0

        return self


@lib.with_doc(_attach_solvent._for_scf.__doc__)
def gostsash_for_scf(mf, solvent_obj=None, dm=None):
    """Attach GOSTSASH solvent model to SCF method."""
    if not isinstance(mf, scf.hf.SCF):
        raise TypeError(
            'GOSTSASH can only be used with SCF methods (RHF/UHF/RKS/UKS). '
            f'Got {mf.__class__.__name__}')
    if solvent_obj is None:
        solvent_obj = GOSTSASH(mf.mol)
    return _attach_solvent._for_scf(mf, solvent_obj, dm)


# Inject GOSTSASH into SCF classes
scf.hf.SCF.GOSTSASH = gostsash_for_scf