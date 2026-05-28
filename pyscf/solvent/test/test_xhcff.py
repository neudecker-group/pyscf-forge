#!/usr/bin/env python

import types
import numpy as np
import pytest
from pyscf import gto, scf
from pyscf.tools import finite_diff
from pyscf.solvent.xhcff import xhcff_external_terms


def test_co2_dimer_xhcff_gradient_vs_qchem():
    """XHCFF external gradient on CO2 dimer matches Q-Chem 6.0 reference.

    Q-Chem settings: PBE/cc-pVDZ, pressure=100000 MPa, scaling=1.0,
    npoints_heavy=302, npoints_hydrogen=302.
    """
    mol = gto.M(
        atom='''O  2.6192991230  -0.0571311942  0.0;
                C  1.6782610262   0.6502025480  0.0;
                O  0.7413912820   1.3674070371  0.0;
                C -1.6782610262  -0.6502025480  0.0;
                O -2.6192991230   0.0571311942  0.0;
                O -0.7413912820  -1.3674070371  0.0''',
        basis='cc-pVDZ',
        unit='Angstrom',
        verbose=0,
    )
    g_ext, _ = xhcff_external_terms(
        mol,
        pressure_mpa=100_000.0,
        npoints=302,
        scaling_factor=1.0,
        rescale_forces=True,
    )
    # Reference: "Gradient from external distort forces" block in Q-Chem output.
    # Rows = xyz, columns = atoms; transposed to (natm, 3) convention.
    ref = np.array([
        [ 0.0666224, -0.0500447,  0.0],
        [ 0.0027262,  0.0021840,  0.0],
        [-0.0625846,  0.0535797,  0.0],
        [-0.0027262, -0.0021840,  0.0],
        [-0.0666224,  0.0500447,  0.0],
        [ 0.0625846, -0.0535797,  0.0],
    ])
    np.testing.assert_allclose(g_ext, ref, rtol=0, atol=1e-7)


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
