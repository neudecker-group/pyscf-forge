# Copyright 2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Standalone X-HCFF helpers (no SCF wrapper class).

This module provides external gradient and external analytic Hessian
contributions for the X-HCFF pressure model.
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


def xhcff_external_terms(mol, pressure_mpa=50_000.0, npoints=302, scaling_factor=1.0,
                         rescale_forces=True):
    """Return (g_ext, h_ext) for X-HCFF.

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
    natm = mol.natm
    coords = mol.atom_coords()

    surf, atom_idx = build_surface(mol, npoints=npoints, scaling_factor=scaling_factor)
    area = surf['area']
    grid = surf['grid_coords']
    nvec = surface_normals(coords, grid, atom_idx)

    p_au = pressure_mpa * MPA_TO_AU
    scale = 1.0 / (scaling_factor ** 2) if rescale_forces else 1.0
    p_eff = p_au * scale

    # External gradient contribution
    sum_all = np.einsum('t,tx->x', area, nvec, optimize=True)
    g_ext = np.zeros((natm, 3))
    for ia in range(natm):
        mask = atom_idx == ia
        sum_i = np.einsum('t,tx->x', area[mask], nvec[mask], optimize=True)
        g_ext[ia] = p_eff * (-sum_i + sum_all / natm)

    # Analytic external Hessian contribution from dA/dR
    _, dA = get_dF_dA(surf, surface_discretization_method='SWIG')
    h_ext = np.zeros((natm, natm, 3, 3))
    for ia in range(natm):
        mask_i = atom_idx == ia
        for ja in range(natm):
            dA_ja = dA[:, ja, :]
            term_i = np.einsum('ta,tb->ab', nvec[mask_i], dA_ja[mask_i], optimize=True)
            term_all = np.einsum('ta,tb->ab', nvec, dA_ja, optimize=True)
            h_ext[ia, ja] = p_eff * (-term_i + term_all / natm)

    # Symmetrize per analytic Hessian formulation
    h_ext = 0.5 * (h_ext + np.transpose(h_ext, (1, 0, 3, 2)))
    return g_ext, h_ext


class WithXHCFFGrad:
    """Mixin that adds X-HCFF external gradient contributions."""

    _keys = {'de_xhcff'}

    def __init__(self, grad_method, xhcff_options):
        self.__dict__.update(grad_method.__dict__)
        self.de_xhcff = None
        self.xhcff_options = dict(xhcff_options)

    def kernel(self, *args, **kwargs):
        de_solute = super().kernel(*args, **kwargs)
        atmlst = kwargs.get('atmlst', self.atmlst)
        g_ext, _ = xhcff_external_terms(self.mol, **self.xhcff_options)
        if atmlst is not None:
            g_ext = g_ext[atmlst]
        self.de_xhcff = g_ext
        self.de = de_solute + g_ext
        return self.de


class WithXHCFFHess:
    """Mixin that adds X-HCFF external Hessian contributions."""

    _keys = {'de_xhcff'}

    def __init__(self, hess_method, xhcff_options):
        self.__dict__.update(hess_method.__dict__)
        self.de_xhcff = None
        self.xhcff_options = dict(xhcff_options)

    def kernel(self, *args, **kwargs):
        de_solute = super().kernel(*args, **kwargs)
        atmlst = kwargs.get('atmlst', self.atmlst)
        _, h_ext = xhcff_external_terms(self.mol, **self.xhcff_options)
        if atmlst is not None:
            h_ext = h_ext[np.ix_(atmlst, atmlst, [0, 1, 2], [0, 1, 2])]
        self.de_xhcff = h_ext
        self.de = de_solute + h_ext
        return self.de


def make_xhcff_grad_object(method, **xhcff_options):
    """Return a gradient object patched with X-HCFF external terms."""
    if isinstance(method, rhf_grad.GradientsBase):
        grad_method = method
    else:
        grad_method = method.nuc_grad_method()
    name = 'XHCFF' + grad_method.__class__.__name__
    return lib.set_class(
        WithXHCFFGrad(grad_method, xhcff_options),
        (WithXHCFFGrad, grad_method.__class__),
        name,
    )


def make_xhcff_hess_object(method, **xhcff_options):
    """Return a Hessian object patched with X-HCFF external terms."""
    if isinstance(method, rhf_hess.HessianBase):
        hess_method = method
    else:
        hess_method = method.Hessian()
    name = 'XHCFF' + hess_method.__class__.__name__
    return lib.set_class(
        WithXHCFFHess(hess_method, xhcff_options),
        (WithXHCFFHess, hess_method.__class__),
        name,
    )


def xhcff_gradients_for_scf(mf, **xhcff_options):
    """Convenience hook: mf.XHCFF_Gradients(**opts)."""
    return make_xhcff_grad_object(mf, **xhcff_options)


def xhcff_hessian_for_scf(mf, **xhcff_options):
    """Convenience hook: mf.XHCFF_Hessian(**opts)."""
    return make_xhcff_hess_object(mf, **xhcff_options)


try:
    from pyscf import scf
    scf.hf.SCF.XHCFF_Gradients = xhcff_gradients_for_scf
    scf.hf.SCF.XHCFF_Hessian = xhcff_hessian_for_scf
except Exception:
    pass
