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
from scipy.optimize import fsolve
import os

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

def xacff_external_terms(mol, pressure_mpa=50_000.0, npoints=302, scaling_factor=1.0,
                         rescale_forces=True, atom_list = None, plane_width = None):
    """Return (g_ext, h_ext) for X-ACFF.

    Args:
        mol: pyscf Mole
        pressure_mpa: Pressure in MPa
        npoints: Lebedev points per atom
        scaling_factor: vdW cavity scaling factor
        rescale_forces: Apply 1/s^2 force rescaling if True

    Returns:
        g_ext: external gradient contribution, shape (natm,3)
        h_ext: external Hessian contribution, shape (natm,natm,3,3)
    """

    if atom_list is None:
        raise ValueError(f"You have not specified a list of three atoms to construct a plane from!")
    if not isinstance(atom_list, list):
        raise ValueError(f"atom_list must be a list of 3 atom indices, got '{atom_list}'")
    if len(atom_list) != 3:
        raise ValueError(f"atom_list must be a list of exactly 3 atom indices, got '{atom_list}'")
    
    atom1 = atom_list[0]
    atom2 = atom_list[1]
    atom3 = atom_list[2]

    a1 = np.array(mol.atom_coord(atom1, unit = 'Bohr'))
    a2 = np.array(mol.atom_coord(atom2, unit = 'Bohr'))
    a3 = np.array(mol.atom_coord(atom3, unit = 'Bohr'))
    normal_normalised, d = create_plane(a1, a2, a3)

    if not isinstance(atom1, int) or not isinstance(atom2, int) or not isinstance(atom3, int):
        raise ValueError(f"atom_list must be a list of exactly 3 atom indices of type int, got '[{atom1}, {atom2}, {atom3}]'")
    if plane_width is None:
        raise ValueError(f"You have to specify the width of the family of planes (in Bohr) to construct the half-shells from!")
    if plane_width <= 0.0:
        raise ValueError(f'plane_width must be positive, got {plane_width} Bohr')

    natm = mol.natm
    coords = mol.atom_coords()

    surf, atom_idx = build_surface(mol, npoints=npoints, scaling_factor=scaling_factor)
    area = surf['area']
    grid = surf['grid_coords']
    nvec = surface_normals(coords, grid, atom_idx)

    distances = np.dot(grid, normal_normalised) - d
    half_width = plane_width / 2.

    slice_mask = np.abs(distances) <= half_width
    print(np.count_nonzero(slice_mask))
    print(len(distances))
    print(f"Screened tess. points: {(np.count_nonzero(slice_mask) / len(distances)) * 100} %")
    if (np.count_nonzero(slice_mask) / len(distances)) > 0.9:
        raise ValueError(f"With the half_width = {half_width} Bohr you have screened away 90% of your tessellation points. \n It is not possible to create reasonable half-shells!")

    anti_slice_mask = np.abs(distances) > half_width
    print(f"half_width = {half_width}")


    def match_average_objective(f):
        """Solves numerically for exponent f, so that the mean of the final array list equals a predefined value"""
        return np.mean((np.abs(distances) / half_width) ** f) - 1.0
        
    f_optimised = fsolve(match_average_objective, x0 = 50.0, full_output= 1)[0]
    print(f"f_optimised = {f_optimised}")

    p_au = pressure_mpa * MPA_TO_AU
    scale = 1.0 / (scaling_factor ** 2) if rescale_forces else 1.0
    p_eff = p_au * scale
        
    p_eff_list = p_eff * (np.abs(distances) / half_width) ** f_optimised
    P_inp_xhcff_list = np.full_like(p_eff_list, p_eff)

    P_inp_xhcff_slice_mask = np.sum(P_inp_xhcff_list[slice_mask])
    print(f"P_inp_xhcff_slice_mask = {P_inp_xhcff_slice_mask} a. u.")
    P_eff_xacff_slice_mask = np.sum(p_eff_list[slice_mask])
    print(f"P_eff_xacff_slice_mask = {P_eff_xacff_slice_mask} a. u.")
    P_inp_xhcff_anti_slice_mask = np.sum(P_inp_xhcff_list[anti_slice_mask])
    print(f"P_inp_xhcff_anti_slice_mask = {P_inp_xhcff_anti_slice_mask} a. u.")
    P_eff_xacff_anti_slice_mask = np.sum(p_eff_list[anti_slice_mask])
    print(f"P_eff_xacff_anti_slice_mask = {P_eff_xacff_anti_slice_mask} a. u.")

    print(f"X-HCFF total sum: {np.sum(P_inp_xhcff_list)} a. u.")
    print(f"X-ACFF total sum: {np.sum(p_eff_list)} a. u.")

    print(np.min(p_eff_list))
    print(np.max(p_eff_list))
    print(f"p_eff_list = {p_eff_list}")
    print(f"P_inp = {p_eff} a.u.")
    print(f"P_eff = {np.mean(p_eff_list)} a.u.")
        

    # External gradient contribution
    sum_all = np.einsum('t,t,tx->x', p_eff_list, area, nvec, optimize=True)
    g_ext = np.zeros((natm, 3))
    
    for ia in range(natm):
        mask = atom_idx == ia
        sum_i = np.einsum('t,t,tx->x', p_eff_list[mask], area[mask], nvec[mask], optimize=True)
        g_ext[ia] = -sum_i + sum_all / natm

    # Analytic external Hessian contribution from dA/dR
    _, dA = get_dF_dA(surf)
    h_ext = np.zeros((natm, natm, 3, 3))

    for ia in range(natm):
        mask_i = atom_idx == ia
        for ja in range(natm):
            dA_ja = dA[:, ja, :]
            term_i = np.einsum('t,ta,tb->ab', p_eff_list[mask_i], nvec[mask_i], dA_ja[mask_i], optimize=True)
            term_all = np.einsum('t,ta,tb->ab', p_eff_list, nvec, dA_ja, optimize=True)
            h_ext[ia, ja] = -term_i + term_all / natm

    # Symmetrize per analytic Hessian formulation
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
