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
GOSTSHYPEnv: GOSTSHYP pressure model with pressure redistributed 
according to a sampled solvent environment.

GOSTSHYP with the pressure on each cavity tessera weighted by a frozen
Gaussian field derived from a sampled solvent environment.

Supports two cavity types:
  - 'vdw': van der Waals surface with scaled Bondi radii
  - 'vdw/occ': Occluded van der Waals surface (crevice-free)

References:
    J. Chem. Theory Comput. 2021, 17, 1, 583-597
    https://doi.org/10.1021/acs.jctc.0c01212

    J. Chem. Theory Comput. 2025, 21, 2, 764-776
    https://doi.org/10.1021/acs.jctc.4c01502
"""
import numpy as np
from functools import cached_property

from pyscf import gto, lib, scf
from pyscf.lib import logger
from pyscf.solvent import _attach_solvent
from pyscf.solvent.pcm import gen_surface, modified_Bondi
from pyscf.solvent.grad.pcm import get_dF_dA
from scipy.spatial import cKDTree
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


def _build_field_trees(field, n_bins=8):
    """Sort the field Gaussians into width classes and build one KD-tree per class.

    Parameters
    ----------
    field : dict
        Solvent field with 'r' (n, 3) centers in Bohr, 'w' (n,) exponents, 'p' (n,) weights.
    n_bins : int
        Number of width classes (quantiles of w).

    Returns
    -------
    trees : list of (ndarray, cKDTree, float)
        Per class: field indices, KD-tree over their centers, smallest exponent in the class.
    """
    fw = field['w']
    edges = np.quantile(fw, np.linspace(0, 1, n_bins + 1))
    cls = np.clip(np.searchsorted(edges, fw, side='right') - 1, 0, n_bins - 1)
    trees = []
    for c in range(n_bins):
        idx = np.nonzero(cls == c)[0]
        if len(idx):
            trees.append((idx, cKDTree(field['r'][idx]), fw[idx].min()))
    return trees


def _field_b_screened(grid_coords, widths, atom_idx, field, trees, deriv=False, eps=1e-20, block_size=16):
    """Field overlap b_j = N_j sum_i p_i <G_j|G_i> and optionally its derivatives, with spatial screening.

    Pairs with a_ji d^2 > -ln(eps) + 5, a_ji = w_j w_i / (w_j + w_i), are skipped. Candidates are found
    with one KD-tree query per block of neighbouring tesserae and width class of the field.

    Parameters
    ----------
    grid_coords : ndarray of shape (ngrids, 3)
        Tessera centers in Bohr.
    widths : ndarray of shape (ngrids,)
        Tessera Gaussian exponents.
    atom_idx : ndarray of shape (ngrids,)
        Atom index of each tessera.
    field : dict
        Solvent field with 'r', 'w', 'p'.
    trees : list
        Output of _build_field_trees(field).
    deriv : bool
        Also return db/dr (eq 17) and db/dw (eq 23).
    eps : float
        Screening threshold (relative contribution of a skipped pair).
    block_size : int
        Number of tesserae per KD-tree query.

    Returns
    -------
    b : ndarray of shape (ngrids,)
    db_dr : ndarray of shape (ngrids, 3), only if deriv
    db_dw : ndarray of shape (ngrids,), only if deriv
    """
    fr, fw, fp = field['r'], field['w'], field['p']
    cut = -np.log(eps) + 5.0

    n_grid = len(widths)
    b = np.zeros(n_grid)
    if deriv:
        db_dr = np.zeros((n_grid, 3))
        db_dw = np.zeros(n_grid)
    for atm in np.unique(atom_idx):
        idx_atm = np.nonzero(atom_idx == atm)[0]
        for blk in np.array_split(idx_atm, max(1, len(idx_atm) // block_size)):
            r_b, w_b = grid_coords[blk], widths[blk]
            center = r_b.mean(axis=0)
            radius = np.sqrt(((r_b - center) ** 2).sum(axis=1).max())
            wb_min = w_b.min()
            sel = []
            for idx_c, tree_c, wf_min in trees:
                mu = wb_min * wf_min / (wb_min + wf_min)
                hits = tree_c.query_ball_point(center, radius + np.sqrt(cut / mu))
                if len(hits):
                    sel.append(idx_c[hits])
            if not sel:
                continue
            sel = np.concatenate(sel)
            rf, wf, pf = fr[sel], fw[sel], fp[sel]

            diff = r_b[:, None, :] - rf[None, :, :]
            d2 = np.einsum('jic,jic->ji', diff, diff)
            inv = 1.0 / (w_b[:, None] + wf[None, :])
            a = w_b[:, None] * wf[None, :] * inv
            ad2 = a * d2
            jj, ii = np.nonzero(ad2 < cut)

            x = np.pi * inv[jj, ii]
            pve = pf[ii] * x * np.sqrt(x) * np.exp(-ad2[jj, ii])
            N = (w_b / np.pi) ** 1.5
            nb = len(blk)
            b[blk] = N * np.bincount(jj, pve, nb)
            if deriv:
                wa = pve * a[jj, ii]
                db_dr[blk] = -2 * N[:, None] * np.stack(
                    [np.bincount(jj, wa * diff[jj, ii, c], nb) for c in range(3)], axis=1)
                r_ = wf[ii] * inv[jj, ii]
                db_dw[blk] = N * np.bincount(jj, pve * (1.5 / w_b[jj] - 1.5 * inv[jj, ii] - r_ * r_ * d2[jj, ii]), nb)
    if deriv:
        return b, db_dr, db_dw
    return b
    

class GOSTSHYPEnv(lib.StreamObject):
    """
    GOSTSHYP pressure solvation model.

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
    solvent_field : dict, optional
        Frozen field {'r' (Bohr), 'w', 'p'} representing the sampled solvent environment
        (crest2gost.build_field), in the aligned frame of the ensemble.
        None gives standard GOSTSHYP.
    area_min : float
        Tesserae with area <= area_min (Bohr^2) are dropped after tessellation
        (default: 0.0). A small positive value (e.g. 1e-5) removes tiny seam
        tesserae whose huge exponents break the gradient integrals.
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
        self.solvent_field = options.get('solvent_field', None)
        self.area_min = options.get('area_min', 0.0)  # Bohr^2

        self.opt_counter = 0
        self.scf_counter = 0

        self.frozen = False
        self.equilibrium_solvation = False
        self.e = None
        self.v = None
        self.amplitudes = None
        self.pressure_au_list = None

        self.build()

    @property
    def pressure_au(self):
        """Pressure in atomic units (Hartree/Bohr^3)."""
        return self.pressure_mpa * MPA_TO_AU

    def build(self, mol=None):
        """Build surface tessellation."""
        if mol is not None:
            self.mol = mol
        mol = self.mol

        rad = self.scaling_factor * modified_Bondi

        if self.cavity == 'vdw/occ':
            rad = rad + self.r_ext
        self.surface_dict = gen_surface(mol, ng=self.npoints, rad=rad)

        # drop tiny seam tesserae: their huge exponents (w = pi ln2 / A) break the integrals
        keep = self.surface_dict['area'] > self.area_min
        self.n_dropped = int((~keep).sum())
        if not keep.all():
            n = [int(keep[p0:p1].sum()) for p0, p1 in self.surface_dict['gslice_by_atom']]
            self.surface_dict['gslice_by_atom'] = [[e - c, e] for c, e in zip(n, np.cumsum(n))]
            for k in ('grid_coords', 'weights', 'charge_exp', 'switch_fun', 'R_vdw', 'norm_vec', 'area'):
                self.surface_dict[k] = self.surface_dict[k][keep]

        if self.cavity == 'vdw/occ':
            r_ext = self.r_ext

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
            self._outer_surface_dict = None
            self._occ_ratio_sq = None
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
        self.N_j = (self.widths / np.pi) ** 1.5  # Gaussian normalization
        self.n_gaussian = len(self.areas)
        self.set_pressure_field()
        self.surface_normals = compute_surface_normals(
            mol.atom_coords(), self.grid_coords, self.atom_idx)

        # Clear cached properties
        self.__dict__.pop('gtilde', None)
        self.__dict__.pop('force_operators', None)

        logger.info(self, 'GOSTSHYP: %d surface Gaussians (cavity=%s)',
                    self.n_gaussian, self.cavity)
        return self

    def set_pressure_field(self):
        self.pressure_au_list = np.full(self.n_gaussian, self.pressure_au)

        if self.solvent_field is not None:
            if getattr(self, '_field_tree_src', None) is not self.solvent_field['r']:
                self._field_trees = _build_field_trees(self.solvent_field)
                self._field_tree_src = self.solvent_field['r']
            if getattr(self, 'field_grad', False):
                self._b, db_dr, db_dw = _field_b_screened(
                    self.grid_coords, self.widths, self.atom_idx, self.solvent_field, self._field_trees, deriv=True)
                self._field_deriv = (self.grid_coords.copy(), self.widths.copy(), db_dr, db_dw)
            else:
                
                self._b = _field_b_screened(
                    self.grid_coords, self.widths, self.atom_idx, self.solvent_field, self._field_trees)
                self._field_deriv = None

            self._b_mean = np.dot(self.areas, self._b) / self.areas.sum()
            self._g = self._b / self._b_mean
            self.pressure_au_list *= self._g

        p = self.pressure_au_list
        p_mean = np.dot(self.areas, p) / self.areas.sum()
        logger.info(self, f"P_inp = {self.pressure_au:.14f} a.u.")
        logger.info(self, f"P_eff = {p_mean:.14f} a.u. (area-weighted mean)")
        logger.info(self, f"P_eff range: {p.min():.6e} - {p.max():.6e} a.u.")

        return self



    def dump_flags(self, verbose=None):
        p = self.pressure_au_list
        logger.info(self, 'pressure (input) = %.1f MPa', self.pressure_mpa)
        logger.info(self, 'P_eff: min %.4e  mean %.4e  max %.4e a.u.',
                    p.min(), np.dot(self.areas, p) / self.areas.sum(), p.max())
        logger.info(self, 'npoints = %d', self.npoints)
        logger.info(self, 'scaling_factor = %.2f', self.scaling_factor)
        logger.info(self, 'cavity = %s', self.cavity)
        if self.cavity == 'vdw/occ':
            logger.info(self, 'r_ext = %.4f Bohr', self.r_ext)
        logger.info(self, 'direct = %s', self.direct)
        logger.info(self, 'n_gaussian = %d', self.n_gaussian)
        logger.info(self, 'area_min = %.1e Bohr^2 (%d tesserae dropped)', self.area_min, self.n_dropped)
        return self

    def check_sanity(self):
        if self.pressure_mpa <= 0:
            raise ValueError(f'pressure_mpa must be positive, got {self.pressure_mpa}')
        if self.scaling_factor <= 0:
            raise ValueError(f'scaling_factor must be positive, got {self.scaling_factor}')
        if self.cavity not in ('vdw', 'vdw/occ'):
            raise ValueError(f"cavity must be 'vdw' or 'vdw/occ', got '{self.cavity}'")
        if self.cavity == 'vdw/occ' and self.r_ext <= 0:
            raise ValueError(f'r_ext must be positive for vdw/occ cavity, got {self.r_ext}')
        if self.write_grid == True and self.output == " ":
            raise ValueError(f'If grid and amplitudes should be written to file, you have to give a file path for the output')
        if self.write_grid == True and not os.path.isdir(self.output):
            raise ValueError(f"The output file path {self.output} you have given does not exist!")
        if self.solvent_field is not None:
            f = self.solvent_field
            if not all(k in f for k in ('r', 'w', 'p')):
                raise ValueError("solvent_field must contain 'r', 'w' and 'p'")
            if not (len(f['r']) == len(f['w']) == len(f['p'])):
                raise ValueError("solvent_field arrays have inconsistent lengths")
        return self

    def kernel(self, dm):
        """Compute GOSTSHYP energy and Fock matrix contribution.

        Parameters
        ----------
        dm : ndarray of shape (nao, nao) or (2, nao, nao)
            Density matrix.

        Returns
        -------
        energy : float
        fock : ndarray of shape (nao, nao)
        """
        if not (isinstance(dm, np.ndarray) and dm.ndim == 2):
            dm = dm[0] + dm[1]

        if self.direct:
            result = self._kernel_direct(dm)
        else:
            result = self._kernel_cached(dm)

        return result

    def _kernel_cached(self, dm):
        """Cached mode: uses precomputed gtilde and force_operators."""
        nao = self.mol.nao_nr()
        dm_flat = dm.ravel(order='F')
        forces = dm_flat @ self.force_operators.reshape(nao * nao, -1, order='F')
        amplitudes = self.pressure_au_list * self.areas / forces
        mask = amplitudes < 0

        #### LOGGING BLOCK FOR NEGATIVE AMPLITUDES; dies currently not exist in kernel_direct and all attributes are only used for potential logging

        self.neg_mask = mask.copy()
        self.neg_area_frac = float(self.areas[mask].sum() / self.areas.sum())
        _abs = float(np.abs(amplitudes).sum())
        self.neg_amp_frac = (float(np.abs(amplitudes[mask]).sum()) / _abs
                             if _abs > 0 else 0.0)

        #######################################################################

        if np.any(mask):
            n_neg = np.count_nonzero(mask)
            logger.warn(self, 'GOSTSHYP: %d/%d (%.1f%%) negative amplitudes removed',
                        n_neg, amplitudes.size, 100.0 * n_neg / amplitudes.size)
            amplitudes[mask] = 0.0
            forces[mask] = 1.0  # avoid division by zero in gradient
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
        logger.info(self, 'GOSTSHYP energy: %.10f', energy)
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

        gmol = fakemol_for_gaussian(self.grid_coords, self.widths, coeffs=self.N_j)
        gmol_p = fakemol_for_gaussian(
            self.grid_coords, self.widths, l=1,
            coeffs=2.0 * self.widths * self.N_j)
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
                logger.warn(self, 'GOSTSHYP: %d/%d (%.1f%%) negative amplitudes removed',
                            n_neg, amplitudes.size, 100.0 * n_neg / amplitudes.size)
                amplitudes[mask] = 0.0
                forces[mask] = 1.0  # avoid division by zero in gradient
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
        logger.info(self, 'GOSTSHYP energy: %.10f', energy)
        self.scf_counter += 1
        return energy, fock

    @cached_property
    def gtilde(self):
        """s-type 3-center overlap integrals [nao, nao, n_gaussian]."""
        mol = self.mol
        gmol = fakemol_for_gaussian(self.grid_coords, self.widths, coeffs=self.N_j)
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
            self.grid_coords, self.widths, l=1,
            coeffs=2.0 * self.widths * self.N_j)
        supermol = mol + gmol_p
        slices = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas + gmol_p.nbas)
        overlap3 = supermol.intor(
            'int3c1e', shls_slice=slices, aosym='s1'
        ).reshape(nao, nao, -1, 3)
        force_ops = np.einsum('bkgc,gc->bkg', overlap3, self.surface_normals,
                              optimize=False)
        return force_ops

    def grad(self, dm):
        """Compute analytical nuclear gradient of GOSTSHYP energy.

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
            result = self._grad_direct(dm)
        else:
            result = self._grad_cached(dm)

        return result

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
        gmol = fakemol_for_gaussian(self.grid_coords, self.widths, coeffs=self.N_j)
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
            coeffs=wgrad_prefs * self.amplitudes * self.N_j)
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
            self.grid_coords, self.widths, l=1, coeffs=coeffs_fop * self.N_j)
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
            self.grid_coords, self.widths, l=3, coeffs=coeffs_f2 * self.N_j)
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

        # Term 4: P_eff derivative
        if self.solvent_field is not None:
            natm = mol.natm

            cache = getattr(self, '_field_deriv', None)
            if (cache is not None and np.array_equal(cache[0], self.grid_coords)
                    and np.array_equal(cache[1], self.widths)):
                db_dr, db_dw = cache[2], cache[3]
            else:
                self.field_grad = True
                _, db_dr, db_dw = _field_b_screened(
                    self.grid_coords, self.widths, self.atom_idx, self.solvent_field, self._field_trees, deriv=True)

            db = np.zeros((natm, 3, self.n_gaussian))   # (natm, 3, ngrids)
            db[self.atom_idx, :, np.arange(self.n_gaussian)] = db_dr    # (natm, 3, ngrids); eq 25 (first part)
            db-= (self.widths / self.areas * db_dw)[None, None, :] * dareas    # (natm, 3, ngrids); eq 25 (second part)

            dAb_sum = np.einsum('acg,g->ac' ,dareas, self._b) + np.einsum('g,acg->ac', self.areas, db)  # (natm, 3); part of eq. 26
            dA_sum = dareas.sum(axis=2)                                                             # (natm, 3)
            dbbar = (dAb_sum - self._b_mean * dA_sum) / self.areas.sum()                            # (natm, 3); eq 26

            dg = (db - self._g[None, None, :] * dbbar[:, :, None]) / self._b_mean                   # (natm, 3, ngrids); eq 15
            dE4 = self.pressure_au * np.einsum('acg,g,g->ac' , dg, self.areas, gtilde_expval / forces, optimize=True)  # (natm, 3); eq 14
        else:
            dE4 = 0.0

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
            N_j_c = self.N_j[chunk_idx]
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
            gmol_s = fakemol_for_gaussian(coords_c, widths_c, coeffs=N_j_c)
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
                coeffs=wgrad_prefs_c * amplitudes_c * N_j_c)
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
                coords_c, widths_c, l=1, coeffs=coeffs_fop_c * N_j_c)
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
                coords_c, widths_c, l=3, coeffs=coeffs_f2_c * N_j_c)
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

        # Term 4: P_eff derivative (no chunking needed)
        if self.solvent_field is not None:
            natm = mol.natm
 
            cache = getattr(self, '_field_deriv', None)
            if (cache is not None and np.array_equal(cache[0], self.grid_coords)
                    and np.array_equal(cache[1], self.widths)):
                db_dr, db_dw = cache[2], cache[3]
            else:
                self.field_grad = True
                _, db_dr, db_dw = _field_b_screened(
                    self.grid_coords, self.widths, self.atom_idx, self.solvent_field, self._field_trees, deriv=True)
            db = np.zeros((natm, 3, self.n_gaussian))   # (natm, 3, ngrids)
            db[self.atom_idx, :, np.arange(self.n_gaussian)] = db_dr    # (natm, 3, ngrids); eq 25 (first part)
            db -= (self.widths / self.areas * db_dw)[None, None, :] * dareas    # (natm, 3, ngrids); eq 25 (second part)

            dAb_sum = np.einsum('acg,g->ac', dareas, self._b) + np.einsum('g,acg->ac', self.areas, db)  # (natm, 3); part of eq. 26
            dA_sum = dareas.sum(axis=2)                                                             # (natm, 3)
            dbbar = (dAb_sum - self._b_mean * dA_sum) / self.areas.sum()                            # (natm, 3); eq 26

            dg = (db - self._g[None, None, :] * dbbar[:, :, None]) / self._b_mean                   # (natm, 3, ngrids); eq 15
            dE4 = self.pressure_au * np.einsum('acg,g,g->ac', dg, self.areas, gtilde_expval / forces, optimize=True)  # (natm, 3); eq 14
        else:
            dE4 = 0.0

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
def gostshyp_env_for_scf(mf, solvent_obj=None, dm=None):
    """Attach GOSTSHYPenv solvent model to SCF method."""
    if not isinstance(mf, scf.hf.SCF):
        raise TypeError(
            'GOSTSHYPEnv can only be used with SCF methods (RHF/UHF/RKS/UKS). '
            f'Got {mf.__class__.__name__}')
    if solvent_obj is None:
        solvent_obj = GOSTSHYPEnv(mf.mol)
    return _attach_solvent._for_scf(mf, solvent_obj, dm)


# Inject GOSTSHYP into SCF classes
scf.hf.SCF.GOSTSHYPEnv = gostshyp_env_for_scf