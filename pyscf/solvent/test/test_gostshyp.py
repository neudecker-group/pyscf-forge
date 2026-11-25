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

import unittest
import numpy as np
from pyscf import gto, scf
from pyscf.solvent.gostshyp import GOSTSHYP, gostshyp_for_scf, compute_surface_normals


def make_hf_mol():
    return gto.M(atom='H 1 0 0; F 2 0 0', basis='6-31g', cart=True, verbose=0)


class TestGOSTSHYP_VDW(unittest.TestCase):
    """Tests with plain vdW cavity."""

    @classmethod
    def setUpClass(cls):
        cls.mol = make_hf_mol()
        cls.gost = GOSTSHYP(cls.mol, options={
            'cavity': 'vdw', 'pressure_mpa': 50_000,
            'npoints': 110, 'scaling_factor': 1.2})
        cls.mf = scf.RHF(cls.mol)
        cls.mf.conv_tol = 1e-12
        cls.mf.conv_tol_grad = 1e-8
        cls.mf = gostshyp_for_scf(cls.mf, cls.gost)
        cls.mf.kernel()

    def test_rhf_energy(self):
        self.assertAlmostEqual(self.mf.e_tot, -99.8941733641653, places=7)

    def test_converged(self):
        self.assertTrue(self.mf.converged)

    def test_fock_symmetry(self):
        dm = self.mf.make_rdm1()
        _, fock = self.gost.kernel(dm)
        np.testing.assert_allclose(fock, fock.T, atol=1e-12)

    def test_gradient(self):
        dm = self.mf.make_rdm1()
        grad = self.gost.grad(dm)
        ref = np.array([[-0.00951946951, 0.0, 0.0],
                        [ 0.00951946951, 0.0, 0.0]])
        np.testing.assert_allclose(grad, ref, atol=1e-7)

    def test_translational_invariance(self):
        dm = self.mf.make_rdm1()
        grad = self.gost.grad(dm)
        np.testing.assert_allclose(grad.sum(axis=0), 0.0, atol=1e-8)

    def test_direct_vs_cached(self):
        dm = self.mf.make_rdm1()
        e_cached, f_cached = self.gost.kernel(dm)

        gost_direct = GOSTSHYP(self.mol, options={
            'cavity': 'vdw', 'pressure_mpa': 50_000,
            'npoints': 110, 'scaling_factor': 1.2, 'direct': True})
        e_direct, f_direct = gost_direct.kernel(dm)

        np.testing.assert_allclose(e_direct, e_cached, atol=1e-12)
        np.testing.assert_allclose(f_direct, f_cached, atol=1e-12)


class TestSurfaceNormals(unittest.TestCase):
    def test_unit_normals(self):
        mol = make_hf_mol()
        gost = GOSTSHYP(mol, options={'cavity': 'vdw'})
        norms = np.linalg.norm(gost.surface_normals, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-14)


class TestGOSTSHYP_UHF(unittest.TestCase):
    def test_uhf_converges(self):
        mol = gto.M(atom='H 1 0 0; F 2 0 0', basis='6-31g', cart=True,
                    verbose=0)
        gost = GOSTSHYP(mol, options={'cavity': 'vdw', 'pressure_mpa': 50_000})
        mf = scf.UHF(mol)
        mf.conv_tol = 1e-10
        mf = gostshyp_for_scf(mf, gost)
        mf.kernel()
        self.assertTrue(mf.converged)


class TestGOSTSHYP_VDW_OCC(unittest.TestCase):
    def test_vdw_occ_converges(self):
        mol = make_hf_mol()
        gost = GOSTSHYP(mol, options={
            'cavity': 'vdw/occ', 'pressure_mpa': 50_000})
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-10
        mf = gostshyp_for_scf(mf, gost)
        mf.kernel()
        self.assertTrue(mf.converged)

    def test_vdw_occ_smaller_area(self):
        """vdW/OCC should have smaller effective areas than plain vdW."""
        mol = make_hf_mol()
        gost_vdw = GOSTSHYP(mol, options={'cavity': 'vdw'})
        gost_occ = GOSTSHYP(mol, options={'cavity': 'vdw/occ'})
        self.assertLess(gost_occ.areas.sum(), gost_vdw.areas.sum())


class TestFiniteDifference(unittest.TestCase):
    def test_gradient_finite_difference(self):
        """Full SCF+GOSTSHYP analytic gradient vs finite_diff."""
        from pyscf.tools import finite_diff
        mol = gto.M(atom='H 1 0 0; F 2 0 0', basis='sto-3g', cart=True,
                    verbose=0)
        gost = GOSTSHYP(mol, options={
            'cavity': 'vdw', 'pressure_mpa': 50_000, 'npoints': 26})
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf = gostshyp_for_scf(mf, gost)
        mf.kernel()

        analytic = mf.Gradients().kernel()
        fd_grad = finite_diff.kernel(mf, displacement=1e-3)
        np.testing.assert_allclose(analytic, fd_grad, atol=1e-5)


class TestDirectGradient(unittest.TestCase):
    def test_direct_gradient_vs_cached(self):
        """Direct gradient must match cached gradient."""
        mol = make_hf_mol()
        opts = {'cavity': 'vdw', 'pressure_mpa': 50_000,
                'npoints': 110, 'scaling_factor': 1.2}

        gost_c = GOSTSHYP(mol, options={**opts, 'direct': False})
        gost_d = GOSTSHYP(mol, options={**opts, 'direct': True})

        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf = gostshyp_for_scf(mf, gost_c)
        mf.kernel()
        dm = mf.make_rdm1()

        gost_c.kernel(dm)
        gost_d.kernel(dm)

        grad_cached = gost_c.grad(dm)
        grad_direct = gost_d.grad(dm)
        np.testing.assert_allclose(grad_direct, grad_cached, atol=1e-12)

    def test_direct_gradient_reference(self):
        """Direct gradient must match known reference values."""
        mol = make_hf_mol()
        gost = GOSTSHYP(mol, options={
            'cavity': 'vdw', 'pressure_mpa': 50_000,
            'npoints': 110, 'scaling_factor': 1.2, 'direct': True})
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf = gostshyp_for_scf(mf, gost)
        mf.kernel()
        dm = mf.make_rdm1()
        gost.kernel(dm)
        grad = gost.grad(dm)
        ref = np.array([[-0.00951946951, 0.0, 0.0],
                        [ 0.00951946951, 0.0, 0.0]])
        np.testing.assert_allclose(grad, ref, atol=1e-7)

    def test_direct_gradient_translational_invariance(self):
        """Direct gradient must sum to zero over atoms."""
        mol = make_hf_mol()
        gost = GOSTSHYP(mol, options={
            'cavity': 'vdw', 'pressure_mpa': 50_000, 'direct': True})
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf = gostshyp_for_scf(mf, gost)
        mf.kernel()
        dm = mf.make_rdm1()
        gost.kernel(dm)
        grad = gost.grad(dm)
        np.testing.assert_allclose(grad.sum(axis=0), 0.0, atol=1e-8)


class TestSCFAttachment(unittest.TestCase):
    def test_mf_gostshyp_method(self):
        """Test that mf.GOSTSHYP() works after import."""
        mol = make_hf_mol()
        mf = scf.RHF(mol)
        mf_sol = mf.GOSTSHYP()
        self.assertTrue(hasattr(mf_sol, 'with_solvent'))

    def test_gostshyp_rejects_post_scf(self):
        from pyscf import mp
        mol = make_hf_mol()
        mf = scf.RHF(mol).run()
        mp2 = mp.MP2(mf)
        with self.assertRaises(TypeError):
            gostshyp_for_scf(mp2)


class TestFiniteDifferenceOCC(unittest.TestCase):
    def test_gradient_finite_difference_occ(self):
        """Full SCF+GOSTSHYP analytic gradient vs finite_diff for vdw/occ cavity."""
        from pyscf.tools import finite_diff
        mol = gto.M(atom='H 1 0 0; F 2 0 0', basis='sto-3g', cart=True, verbose=0)
        gost = GOSTSHYP(mol, options={
            'cavity': 'vdw/occ', 'pressure_mpa': 50_000, 'npoints': 110})
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf = gostshyp_for_scf(mf, gost)
        mf.kernel()
        analytic = mf.Gradients().kernel()
        fd_grad = finite_diff.kernel(mf, displacement=1e-3)
        np.testing.assert_allclose(analytic, fd_grad, atol=1e-5)


class TestSphericalHarmonics(unittest.TestCase):
    def test_gradient_spherical_basis(self):
        """Gradient with spherical harmonic basis (exercises c2s transform)."""
        from pyscf.tools import finite_diff
        mol = gto.M(atom='H 1 0 0; F 2 0 0', basis='6-31g', cart=False, verbose=0)
        gost = GOSTSHYP(mol, options={
            'cavity': 'vdw', 'pressure_mpa': 50_000, 'npoints': 26})
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf = gostshyp_for_scf(mf, gost)
        mf.kernel()
        analytic = mf.Gradients().kernel()
        fd_grad = finite_diff.kernel(mf, displacement=1e-3)
        np.testing.assert_allclose(analytic, fd_grad, atol=1e-5)


class TestMultiAtom(unittest.TestCase):
    def test_water_gradient(self):
        """Gradient for a non-linear molecule (exercises np.add.at accumulation)."""
        from pyscf.tools import finite_diff
        mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
                    basis='sto-3g', cart=True, verbose=0)
        gost = GOSTSHYP(mol, options={
            'cavity': 'vdw', 'pressure_mpa': 50_000, 'npoints': 26})
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf = gostshyp_for_scf(mf, gost)
        mf.kernel()
        analytic = mf.Gradients().kernel()
        fd_grad = finite_diff.kernel(mf, displacement=1e-3)
        np.testing.assert_allclose(analytic, fd_grad, atol=1e-5)


class TestNegativeAmplitudeMasking(unittest.TestCase):
    """Tight cavity triggers negative amplitudes; verify they are masked."""

    def test_no_negative_amplitudes(self):
        mol = make_hf_mol()
        gost = GOSTSHYP(mol, options={
            'cavity': 'vdw', 'pressure_mpa': 50_000,
            'npoints': 110, 'scaling_factor': 0.5})
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf = gostshyp_for_scf(mf, gost)
        mf.kernel()
        self.assertTrue(mf.converged)
        self.assertTrue(np.all(mf.with_solvent.amplitudes >= 0))

    def test_gradient_finite_difference_tight_cavity(self):
        from pyscf.tools import finite_diff
        mol = gto.M(atom='H 1 0 0; F 2 0 0', basis='sto-3g', cart=True,
                    verbose=0)
        gost = GOSTSHYP(mol, options={
            'cavity': 'vdw', 'pressure_mpa': 50_000,
            'npoints': 26, 'scaling_factor': 0.5})
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf = gostshyp_for_scf(mf, gost)
        mf.kernel()
        analytic = mf.Gradients().kernel()
        fd_grad = finite_diff.kernel(mf, displacement=1e-3)
        np.testing.assert_allclose(analytic, fd_grad, atol=1e-5)


class TestReset(unittest.TestCase):
    def test_reset_rebuilds_surface(self):
        """reset() clears cached properties and rebuilds for new geometry."""
        mol1 = gto.M(atom='H 1 0 0; F 2 0 0', basis='sto-3g', cart=True, verbose=0)
        gost = GOSTSHYP(mol1, options={'cavity': 'vdw', 'pressure_mpa': 50_000})
        mf = scf.RHF(mol1)
        mf.conv_tol = 1e-12
        mf = gostshyp_for_scf(mf, gost)
        mf.kernel()
        e1 = mf.e_tot

        mol2 = gto.M(atom='H 1 0 0; F 2.2 0 0', basis='sto-3g', cart=True, verbose=0)
        mf.reset(mol2)
        mf.kernel()
        e2 = mf.e_tot
        self.assertNotAlmostEqual(e1, e2, places=5)
        self.assertTrue(mf.converged)


class TestSmallAreaPoints(unittest.TestCase):
    """Surface points with tiny area made the analytic gradient wrong.

    In this randomly distorted cyclopentadiene (sigma = 0.1 Angstrom, numpy seed 7)
    the vdW/OCC surface has a point of area 3.7e-15 Bohr^2 on atom 10 (Gaussian
    width ~6e14). Without removing such points, the analytic gradient of atoms 4
    and 10 is wrong by 4 to 32 Eh/Bohr (random numerical noise, ~3e-15/A to
    1e-13/A), so symmetric test molecules (smallest area >= 1e-9) never showed it.
    Removal is optional
    in GOSTSHYP ('min_area', default None = plain gen_surface as in the theory).
    """
    OPTS = {'npoints': 110, 'pressure_mpa': 10_000, 'min_area': 1e-7}
    # Bohr
    CP_DISTORTED = [
        ['C', [0.000232465294, 2.280662373296, -0.694311429053]],
        ['C', [-0.168297466402, -2.310127974695, -0.829900922483]],
        ['C', [0.011365493705, 1.636543499388, 1.664431744167]],
        ['C', [-0.117252762783, -1.290712791267, 1.824887166129]],
        ['C', [0.019920406023, -0.175832977216, -2.273199142788]],
        ['H', [0.131393261106, 3.903377739341, -1.371490610569]],
        ['H', [-0.359279028002, -4.401084789592, -1.633051246248]],
        ['H', [-0.044425785205, 2.368309379153, 3.452768558770]],
        ['H', [0.029621662345, -2.643146880855, 2.925908366738]],
        ['H', [1.561160785777, -0.009165350359, -3.493478296591]],
        ['H', [-1.952112742638, -0.090282284682, -3.699803898210]]]

    @staticmethod
    def energy(atom):
        mol = gto.M(atom=atom, unit='Bohr', basis='sto-3g', verbose=0)
        mf = gostshyp_for_scf(scf.RHF(mol), GOSTSHYP(mol, options=dict(TestSmallAreaPoints.OPTS)))
        mf.conv_tol = 1e-12
        e = mf.kernel()
        assert mf.converged
        return mf, e

    def test_no_filtering_by_default(self):
        """Default: plain gen_surface, the 3.7e-15 Bohr^2 point is kept."""
        mol = gto.M(atom=self.CP_DISTORTED, unit='Bohr', basis='sto-3g', verbose=0)
        gost = GOSTSHYP(mol, options={'npoints': 110, 'pressure_mpa': 10_000})
        self.assertIsNone(gost.min_area)
        self.assertEqual(gost.n_removed_points, 0)
        self.assertLess(gost.areas.min(), 1e-14)

    def test_min_area_validation(self):
        mol = gto.M(atom=self.CP_DISTORTED, unit='Bohr', basis='sto-3g', verbose=0)
        for bad in (-1e-7, '1e-7', True):
            with self.assertRaises(ValueError):
                GOSTSHYP(mol, options={'npoints': 110, 'min_area': bad})

    def test_small_points_removed(self):
        mol = gto.M(atom=self.CP_DISTORTED, unit='Bohr', basis='sto-3g', verbose=0)
        gost = GOSTSHYP(mol, options=dict(self.OPTS))
        self.assertGreater(gost.n_removed_points, 0)
        self.assertGreater(gost.areas.min(), self.OPTS['min_area'])
        self.assertEqual(gost.surface_dict['gslice_by_atom'][-1][1], gost.n_gaussian)
        self.assertEqual(len(gost._outer_surface_dict['area']), gost.n_gaussian)

    def test_gradient_atom10(self):
        mf, _ = self.energy(self.CP_DISTORTED)
        ana = mf.Gradients().kernel()
        h = 1e-4
        for c in range(3):
            ap = [[s, list(x)] for s, x in self.CP_DISTORTED]; ap[10][1][c] += h
            am = [[s, list(x)] for s, x in self.CP_DISTORTED]; am[10][1][c] -= h
            num = (self.energy(ap)[1] - self.energy(am)[1]) / (2 * h)
            self.assertAlmostEqual(ana[10, c], num, delta=1e-7)


if __name__ == '__main__':
    unittest.main()
