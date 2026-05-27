#!/usr/bin/env python

import types
import numpy as np
import pytest
from scipy.optimize import brentq
from pyscf import gto, scf
from pyscf.tools import finite_diff
from pyscf.solvent.xhcff import xhcff_external_terms


def _hf_total_grad_fz(r_ang, pressure_mpa=0.0, npoints=110, scaling=1.2):
    mol = gto.M(
        atom=f'H 0 0 0; F 0 0 {r_ang}',
        basis='sto-3g',
        unit='Angstrom',
        verbose=0,
    )
    mf = scf.RHF(mol).run(conv_tol=1e-10)
    grad = mf.Gradients().kernel()
    if pressure_mpa > 0:
        g_ext, _ = xhcff_external_terms(
            mol,
            pressure_mpa=pressure_mpa,
            npoints=npoints,
            scaling_factor=scaling,
            rescale_forces=True,
        )
        grad = grad + g_ext
    return grad[1, 2]


def _optimize_hf_bond(pressure_mpa=0.0):
    return brentq(lambda r: _hf_total_grad_fz(r, pressure_mpa=pressure_mpa), 0.8, 1.1)


def test_hf_is_compressed_vs_vacuum():
    r_vac = _optimize_hf_bond(pressure_mpa=0.0)
    r_press = _optimize_hf_bond(pressure_mpa=50_000.0)
    assert r_press < r_vac
    assert (r_vac - r_press) < 0.05


def test_scf_convenience_methods_shape_and_pyscf_symmetry():
    mol = gto.M(atom='H 0 0 0; F 0 0 1.0', basis='sto-3g', unit='Angstrom', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-10)

    g = mf.XHCFF_Gradients(
        pressure_mpa=10_000.0,
        npoints=110,
        scaling_factor=1.2,
        rescale_forces=True,
    ).kernel()
    h = mf.XHCFF_Hessian(
        pressure_mpa=10_000.0,
        npoints=110,
        scaling_factor=1.2,
        rescale_forces=True,
    ).kernel()

    assert g.shape == (mol.natm, 3)
    assert h.shape == (mol.natm, mol.natm, 3, 3)

    # PySCF Hessian symmetry convention: H[i,j,a,b] = H[j,i,b,a]
    assert np.linalg.norm(h - np.transpose(h, (1, 0, 3, 2))) < 1e-9


@pytest.mark.parametrize('rescale_forces', [False, True])
def test_analytic_external_hessian_vs_finite_difference(rescale_forces):
    mol = gto.M(
        atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
        basis='sto-3g',
        unit='Angstrom',
        verbose=0,
    )
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    opts = dict(
        pressure_mpa=10_000.0,
        npoints=110,
        scaling_factor=1.2,
        rescale_forces=rescale_forces,
    )

    _, h_ana = xhcff_external_terms(mol, **opts)

    # Use PySCF finite-diff utility on an external-gradient-only Gradients object
    g_obj = mf.Gradients()
    g_obj.base.conv_tol = 1e-14

    def _kernel_external(self, *args, **kwargs):
        return xhcff_external_terms(self.mol, **opts)[0]

    g_obj.kernel = types.MethodType(_kernel_external, g_obj)
    h_fd = finite_diff.kernel(g_obj, displacement=1e-4)

    # Match analytic Hessian convention used by xhcff_external_terms
    h_fd = 0.5 * (h_fd + np.transpose(h_fd, (1, 0, 3, 2)))

    np.testing.assert_allclose(h_ana, h_fd, rtol=0.0, atol=1e-7)
