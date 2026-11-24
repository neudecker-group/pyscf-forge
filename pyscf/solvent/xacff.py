# Copyright 2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Standalone X-ACFF helpers (no SCF wrapper class).

This module provides external gradient and external analytic Hessian
contributions for the eXtended Anisotropic Compression Force Field (X-ACFF) pressure model.
"""

import numpy as np
from pyscf import lib
from pyscf.grad import rhf as rhf_grad
from pyscf.hessian import rhf as rhf_hess
from pyscf.solvent.pcm import gen_surface, modified_Bondi
from pyscf.solvent.grad.pcm import get_dF_dA

# 1 MPa in Eh/Bohr^3
MPA_TO_AU = 3.3989309735473356e-08


def build_surface(mol, npoints=302, scaling_factor=1.0):
    rad = scaling_factor * modified_Bondi
    surf = gen_surface(mol, ng=npoints, rad=rad)
    atom_idx = np.zeros(len(surf['area']), dtype=np.int32)
    for ia, (p0, p1) in enumerate(surf['gslice_by_atom']):
        atom_idx[p0:p1] = ia
    return surf, atom_idx


def surface_normals(atom_coords, grid_coords, atom_idx):
    dr = atom_coords[atom_idx] - grid_coords
    return dr / np.linalg.norm(dr, axis=1, keepdims=True)

def create_plane(p1, p2, p3):
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
    v1 = p2 - p1
    v2 = p3 - p1 

    # cross product of v1 and v2 is orthogonal to them and the normal vector of the plane
    normal = np.cross(v1, v2)
    #print(f"normal = {normal}")

    # Normalisation of the normal vector for exact euclidean distances
    normal_length = np.linalg.norm(normal)
    if np.abs(normal_length) < 1e-5:
        raise ValueError("The three atoms lay on a straight line, therefore they cannot form a plane!")
    # normal_normalised directs from the origin in direction of the plane
    normal_normalised = normal / normal_length
 
    # insert point 1 into plane equation: d = n_0 · p1
    # d is distance of plane to origin
    d = np.dot(normal_normalised, p1)

    return normal_normalised, d


def check_f_exponent(f_exponent):
    """Validate the exponent of the anisotropic pressure distribution and return it as float."""
    # no default on purpose: f has not been benchmarked yet
    if f_exponent is None:
        raise ValueError("X-ACFF requires the option 'f_exponent' (exponent of the anisotropic "
                         "pressure distribution, e.g. f_exponent=4). There is no default value.")
    if isinstance(f_exponent, bool) or not isinstance(f_exponent, (int, float)):
        raise ValueError(f"f_exponent must be a number, got '{f_exponent}'")
    if f_exponent < 1.0:
        raise ValueError(f"f_exponent must be >= 1, got {f_exponent}")
    return float(f_exponent)


def dnormal_dX(natm, atom_list, p1, p2, p3):
    """Derivative of the unit plane normal n0 with respect to all nuclear coordinates.

    Only the three plane atoms contribute. Returns shape (natm, 3, 3): [atom, cartesian, n0 component].
    """
    n = np.cross(p2 - p1, p3 - p1)
    n_norm = np.linalg.norm(n)
    n0 = n / n_norm
    dn0 = np.zeros((natm, 3, 3))
    for c in range(3):
        ec = np.zeros(3)
        ec[c] = 1.0
        dn_dx2 = np.cross(ec, p3 - p1)
        dn_dx3 = np.cross(p2 - p1, ec)
        dn_dx1 = -dn_dx2 - dn_dx3
        for ia, dn in zip(atom_list, (dn_dx1, dn_dx2, dn_dx3)):
            dn0[ia, c] = (dn - np.dot(n0, dn) * n0) / n_norm
    return dn0


def dpressure_dX(p_eff, x_pow_f, mean_x_pow_f, base_fraction, distances, half_width,
                 f_exponent, area, dareas, grid, atom_idx, atom_list, p1, p2, p3, normal):
    """Derivative of p_t = P * x_t^f / M, M = sum(A x^f) / sum(A), with respect to all
    nuclear coordinates lambda (same as GOSTSASH Term 4):

        dp_t/dlambda = (P / M) * [u_t - x_t^f * (dM/dlambda) / M],
        u_t = (f / half_width) * sign(d_t) * x_t^(f-1) * dd_t/dlambda,
        dd_t/dlambda = delta(B, A(t)) n0_c - delta(B, atom1) n0_c + dn0/dlambda . (r_t - x1),
        dM/dlambda = [sum_t (dA_t x_t^f + A_t u_t) - M sum_t dA_t] / sum_t A_t

    p_eff: input pressure P (after force rescaling), dareas: shape (natm, 3, ngrid).
    Returns shape (natm, 3, ngrid).
    """
    natm = dareas.shape[0]
    dn0 = dnormal_dX(natm, atom_list, p1, p2, p3)
    # dd_t/dlambda, (natm, 3, ngrid): grid points move rigidly with their atom,
    # the plane moves with its three atoms
    ddt = np.einsum('acn,gn->acg', dn0, grid - p1)
    for ia in range(natm):
        ddt[ia][:, atom_idx == ia] += normal[:, None]
    ddt[atom_list[0]] -= normal[:, None]

    u = ddt * ((f_exponent / half_width) * np.sign(distances) * base_fraction ** (f_exponent - 1))
    M = mean_x_pow_f
    dM = (np.einsum('acg,g->ac', dareas, x_pow_f) + np.einsum('g,acg->ac', area, u)
          - M * dareas.sum(axis=2)) / area.sum()
    return (p_eff / M) * (u - x_pow_f * (dM / M)[:, :, None])


def xacff_external_terms(mol, pressure_mpa=50_000.0, npoints=302, scaling_factor=1.0,
                         rescale_forces=True, atom_list = None, plane_width = None,
                         f_exponent = None, symmetrize=True):
    """Return (g_ext, h_ext) for X-ACFF.

    Args:
        mol: pyscf Mole
        pressure_mpa: Pressure in MPa
        npoints: Lebedev points per atom
        scaling_factor: vdW cavity scaling factor
        rescale_forces: Apply 1/s^2 force rescaling if True
        atom_list: three atom indices defining the plane
        plane_width: width of the band around the plane in Bohr
        f_exponent: exponent f of the anisotropic pressure distribution
            (required, no default, >= 1). Each surface point t gets
            p_t = P * x_t**f / M with x_t = |d_t| / (plane_width / 2) and the
            area-weighted mean M = sum(A_t x_t**f) / sum(A_t), so the total load
            sum(p_t A_t) equals P * sum(A_t) as in X-HCFF.
        symmetrize: return 0.5 (H + H^T) (default). The X-ACFF force is not the
            gradient of an energy, so its Jacobian is not symmetric in general;
            False returns the plain Jacobian d g_ext / dX.

    Returns:
        g_ext: external gradient contribution, shape (natm,3)
        h_ext: external Hessian contribution, shape (natm,natm,3,3)
    """
    f_exponent = check_f_exponent(f_exponent)

    if atom_list is None:
        raise ValueError(f"You have not specified a list of three atoms to construct a plane from!")
    if not isinstance(atom_list, list):
        raise ValueError(f"atom_list must be a list of 3 atom indices, got '{atom_list}'")
    if len(atom_list) != 3:
        raise ValueError(f"atom_list must be a list of exactly 3 atom indices, got '{atom_list}'")
    if not isinstance(atom_list[0], int) or not isinstance(atom_list[1], int) or not isinstance(atom_list[2], int):
        raise ValueError(f"atom_list must be a list of exactly 3 atom indices of type int, got '[{atom_list[0]}, {atom_list[1]}, {atom_list[2]}]'")

    atom1 = atom_list[0]
    atom2 = atom_list[1]
    atom3 = atom_list[2]

    if atom1 == atom2 or atom1 == atom3 or atom2 == atom3:
        raise ValueError(f"The three atom indices should be different atoms, got: [{atom1}, {atom2}, {atom3}]!")
    if 0 <= atom1 < mol.natm:
        a1 = np.array(mol.atom_coord(atom1, unit = 'Bohr'))
    else:
        raise ValueError(f"The atom index {atom1} does not exist in this molecule!")
    if 0 <= atom2 < mol.natm:
        a2 = np.array(mol.atom_coord(atom2, unit = 'Bohr'))
    else:
        raise ValueError(f"The atom index {atom2} does not exist in this molecule!")
    if 0 <= atom3 < mol.natm:
        a3 = np.array(mol.atom_coord(atom3, unit = 'Bohr'))
    else:
        raise ValueError(f"The atom index {atom3} does not exist in this molecule!")
    normal_normalised, d = create_plane(a1, a2, a3)

    natm = mol.natm
    coords = mol.atom_coords()

    surf, atom_idx = build_surface(mol, npoints=npoints, scaling_factor=scaling_factor)
    area = surf['area']
    grid = surf['grid_coords']
    nvec = surface_normals(coords, grid, atom_idx)

    distances = np.dot(grid, normal_normalised) - d

    if plane_width is None:
        raise ValueError(f"You have to specify the width of the family of planes (in Bohr) to construct the half-shells from!")
    if plane_width <= 0.0:
        raise ValueError(f'plane_width must be positive, got {plane_width} Bohr')
        
    half_width = plane_width / 2.

    slice_mask = np.abs(distances) <= half_width
    print(f"Tess. points in cylindrical part: {(np.count_nonzero(slice_mask) / len(distances)) * 100:.3f} %")
    if np.count_nonzero(slice_mask) == len(distances):
        raise ValueError(f"It is not possible to create reasonable half-shells! \nPlease try again using a smaller width than {plane_width} Bohr!")
    print(f"half_width = {half_width:.3f} Bohr")


    p_au = pressure_mpa * MPA_TO_AU
    scale = 1.0 / (scaling_factor ** 2) if rescale_forces else 1.0
    p_eff = p_au * scale

    # p_t = P * x_t^f / M with the area-weighted mean M = sum(A x^f) / sum(A):
    # monotonic in the distance to the plane, total load sum(p A) = P sum(A) as in X-HCFF
    base_fraction = np.abs(distances) / half_width
    x_pow_f = base_fraction ** f_exponent
    mean_x_pow_f = np.dot(area, x_pow_f) / np.sum(area)
    # band points (x <= 1) only get less than P if M > 1
    if mean_x_pow_f <= 1.0:
        raise ValueError(f"The pressure inside the plane band would not be reduced (<x^f> = {mean_x_pow_f:.4f} <= 1)! \nPlease try again using a smaller width than {plane_width} Bohr!")
    p_eff_list = p_eff * x_pow_f / mean_x_pow_f

    print(f"f_exponent = {f_exponent}")
    print(f"<x^f> = {mean_x_pow_f:.6e}")
    print(f"p_t = P_inp at {mean_x_pow_f ** (1.0 / f_exponent) * half_width:.3f} Bohr from the plane")
    print(f"P_inp = {p_eff:.14f} a.u.")
    print(f"P_eff = {np.dot(area, p_eff_list) / np.sum(area):.14f} a.u. (area-weighted mean)")

    # External gradient contribution
    sum_all = np.einsum('t,t,tx->x', p_eff_list, area, nvec, optimize=True)
    g_ext = np.zeros((natm, 3))
    
    for ia in range(natm):
        mask = atom_idx == ia
        sum_i = np.einsum('t,t,tx->x', p_eff_list[mask], area[mask], nvec[mask], optimize=True)
        g_ext[ia] = -sum_i + sum_all / natm

    # Analytic external Hessian: derivative of q_t = p_t A_t n_t. The normals
    # n_t = (X_A(t) - r_t) / |X_A(t) - r_t| do not change (r_t moves rigidly with
    # its atom), so d q_t = n_t (p_t dA_t + A_t dp_t).
    _, dA = get_dF_dA(surf)                       # (ngrid, natm, 3)
    dareas = dA.transpose(1, 2, 0)                # (natm, 3, ngrid)
    dp = dpressure_dX(p_eff, x_pow_f, mean_x_pow_f, base_fraction, distances, half_width,
                      f_exponent, area, dareas, grid, atom_idx, atom_list, a1, a2, a3,
                      normal_normalised)
    dq_scalar = p_eff_list * dareas + area * dp   # (natm, 3, ngrid): d(p_t A_t)/dX_jb
    h_ext = np.zeros((natm, natm, 3, 3))

    for ia in range(natm):
        mask_i = atom_idx == ia
        for ja in range(natm):
            term_i = np.einsum('ta,bt->ab', nvec[mask_i], dq_scalar[ja][:, mask_i], optimize=True)
            term_all = np.einsum('ta,bt->ab', nvec, dq_scalar[ja], optimize=True)
            h_ext[ia, ja] = -term_i + term_all / natm

    # Symmetrize per analytic Hessian formulation
    if symmetrize:
        h_ext = 0.5 * (h_ext + np.transpose(h_ext, (1, 0, 3, 2)))
    return g_ext, h_ext


class WithXACFFGrad:
    """Mixin that adds X-ACFF external gradient contributions."""

    _keys = {'de_xacff'}

    def __init__(self, grad_method, xacff_options):
        self.__dict__.update(grad_method.__dict__)
        self.de_xacff = None
        self.xacff_options = dict(xacff_options)

    def kernel(self, *args, **kwargs):
        de_solute = super().kernel(*args, **kwargs)
        atmlst = kwargs.get('atmlst', self.atmlst)
        g_ext, _ = xacff_external_terms(self.mol, **self.xacff_options)
        if atmlst is not None:
            g_ext = g_ext[atmlst]
        self.de_xacff = g_ext
        self.de = de_solute + g_ext
        return self.de


class WithXACFFHess:
    """Mixin that adds X-ACFF external Hessian contributions."""

    _keys = {'de_xacff'}

    def __init__(self, hess_method, xacff_options):
        self.__dict__.update(hess_method.__dict__)
        self.de_xacff = None
        self.xacff_options = dict(xacff_options)

    def kernel(self, *args, **kwargs):
        de_solute = super().kernel(*args, **kwargs)
        atmlst = kwargs.get('atmlst', self.atmlst)
        _, h_ext = xacff_external_terms(self.mol, **self.xacff_options)
        if atmlst is not None:
            h_ext = h_ext[np.ix_(atmlst, atmlst, [0, 1, 2], [0, 1, 2])]
        self.de_xacff = h_ext
        self.de = de_solute + h_ext
        return self.de


def make_xacff_grad_object(method, **xacff_options):
    """Return a gradient object patched with X-ACFF external terms."""
    if isinstance(method, rhf_grad.GradientsBase):
        grad_method = method
    else:
        grad_method = method.nuc_grad_method()
    name = 'XACFF' + grad_method.__class__.__name__
    return lib.set_class(
        WithXACFFGrad(grad_method, xacff_options),
        (WithXACFFGrad, grad_method.__class__),
        name,
    )


def make_xacff_hess_object(method, **xhcff_options):
    """Return a Hessian object patched with X-ACFF external terms."""
    if isinstance(method, rhf_hess.HessianBase):
        hess_method = method
    else:
        hess_method = method.Hessian()
    name = 'XACFF' + hess_method.__class__.__name__
    return lib.set_class(
        WithXACFFHess(hess_method, xhcff_options),
        (WithXACFFHess, hess_method.__class__),
        name,
    )


def xacff_gradients_for_scf(mf, **xacff_options):
    """Convenience hook: mf.XACFF_Gradients(**opts)."""
    return make_xacff_grad_object(mf, **xacff_options)


def xacff_hessian_for_scf(mf, **xacff_options):
    """Convenience hook: mf.XHCFF_Hessian(**opts)."""
    return make_xacff_hess_object(mf, **xacff_options)


try:
    from pyscf import scf
    scf.hf.SCF.XACFF_Gradients = xacff_gradients_for_scf
    scf.hf.SCF.XACFF_Hessian = xacff_hessian_for_scf
except Exception:
    pass
