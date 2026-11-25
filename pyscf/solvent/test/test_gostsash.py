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

'''
Tests for the GOSTSASH anisotropic pressure distribution
p_t = P * |d_t|^f / M, M = sum(A |d|^f) / sum(A), d_t = distance to the plane,
with a fixed exponent f (no plane width: it would cancel out of p_t).
'''

import unittest
from unittest import mock
import numpy as np
from pyscf import gto, scf
from pyscf.solvent.grad.pcm import get_dF_dA
from pyscf.solvent.gostsash import GOSTSASH, gostsash_for_scf
from pyscf.solvent.gostshyp import MIN_SURFACE_AREA

# Bohr
NH3 = [['N', [0.000, 0.000, 0.227]],
       ['H', [0.000, 1.776, -0.510]],
       ['H', [1.531, -0.888, -0.510]],
       ['H', [-1.531, -0.888, -0.548]]]

# Angstrom; planar ring in x = 0, CH2 hydrogens in y = 0
CP = '''C 0.0000 1.1770 -0.3400
C 0.0000 -1.1770 -0.3400
C 0.0000 0.7320 0.9300
C 0.0000 -0.7320 0.9300
C 0.0000 0.0000 -1.2000
H 0.0000 2.2000 -0.6800
H 0.0000 -2.2000 -0.6800
H 0.0000 1.3800 1.8000
H 0.0000 -1.3800 1.8000
H 0.8800 0.0000 -1.8600
H -0.8800 0.0000 -1.8600'''

# Bohr; NH3 (N 0 0 0.12, H 0 0.94 -0.27, H +-0.81 -0.47 -0.27 Angstrom) randomly
# distorted (sigma = 0.1 Angstrom, numpy seed 7): no symmetry left, so the gradient
# test does not profit from error cancellation between equivalent atoms
NH3_DISTORTED = [
    ['N', [0.000232465294, 0.056454724683, 0.174962588247]],
    ['H', [-0.168297466402, 1.690422231010, -0.697620093764]],
    ['H', [1.542043654603, -0.634907302339, -0.603239605311]],
    ['H', [-1.647930923680, -0.795604546631, -0.442784183349]]]

# Bohr; CP randomly distorted (sigma = 0.1 Angstrom, numpy seed 7). Its vdW/OCC
# surface contains a point of area 3.7e-15 Bohr^2 on atom 10. Without removing
# such points (MIN_SURFACE_AREA), the analytic gradient of atoms 4 and 10 is wrong
# by random numerical noise (GOSTSHYP 4-32 Eh/Bohr; GOSTSASH scaled by p_t/P).
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


def make_options(atom_list, f, **kw):
    opts = dict(npoints=110, atom_list=atom_list, pressure_mpa=10000, f_exponent=f)
    opts.update(kw)
    return opts


def make_mol(atom, unit='Bohr'):
    return gto.M(atom=atom, unit=unit, basis='sto-3g', verbose=0)


def displaced(mol, i, c, h):
    coords = mol.atom_coords().copy()
    coords[i, c] += h
    atom = [[mol.atom_symbol(k), coords[k]] for k in range(mol.natm)]
    return make_mol(atom, unit='Bohr')


def area_derivatives(g):
    """dA_t/dX of the areas used in the energy, shape (natm, 3, ngrid)."""
    if g._outer_surface_dict is not None:
        _, dA = get_dF_dA(g._outer_surface_dict)
        return dA.transpose(1, 2, 0) * g._occ_ratio_sq
    _, dA = get_dF_dA(g.surface_dict)
    return dA.transpose(1, 2, 0)


class TestGOSTSASHOptions(unittest.TestCase):

    def test_f_exponent_required(self):
        mol = make_mol(NH3)
        opts = make_options([0, 1, 2], 4)
        del opts['f_exponent']
        with self.assertRaises(ValueError):
            GOSTSASH(mol, opts)

    def test_f_exponent_below_one(self):
        with self.assertRaises(ValueError):
            GOSTSASH(make_mol(NH3), make_options([0, 1, 2], 0.5))

    def test_f_exponent_not_a_number(self):
        with self.assertRaises(ValueError):
            GOSTSASH(make_mol(NH3), make_options([0, 1, 2], '4'))

    def test_plane_width_removed(self):
        with self.assertRaises(ValueError):
            GOSTSASH(make_mol(NH3), make_options([0, 1, 2], 4, plane_width=1.0))


class TestGOSTSASHPressureDistribution(unittest.TestCase):

    def test_mean_pressure(self):
        """Area-weighted mean pressure = P, i.e. the total load sum(p A) equals P sum(A)."""
        for f in (1, 2, 4, 8):
            g = GOSTSASH(make_mol(CP, unit='Ang'), make_options([0, 1, 2], f))
            mean = np.dot(g.areas, g.pressure_au_list) / np.sum(g.areas)
            self.assertAlmostEqual(mean / g.pressure_au, 1.0, places=12)

    def test_monotonic_in_distance(self):
        g = GOSTSASH(make_mol(CP, unit='Ang'), make_options([0, 1, 2], 4))
        order = np.argsort(g.abs_distances)
        self.assertTrue(np.all(np.diff(g.pressure_au_list[order]) >= 0))
        inner = g.abs_distances < g.crossover_distance
        self.assertTrue(np.all(g.pressure_au_list[inner] < g.pressure_au))
        self.assertTrue(np.all(g.pressure_au_list[~inner] >= g.pressure_au))
        self.assertGreater(np.max(g.pressure_au_list), g.pressure_au)

    def test_dpressure_dX_finite_difference(self):
        """Analytic dp_t/dX against central finite differences (was test_dp.py)."""
        h = 1e-5
        for atom, unit, al in ((NH3, 'Bohr', [0, 1, 2]), (CP, 'Ang', [2, 5, 6])):
            mol = make_mol(atom, unit=unit)
            for f in (1, 2, 4, 8):
                g = GOSTSASH(mol, make_options(al, f))
                ana = g.compute_dpressure_dX(area_derivatives(g))
                for i in range(mol.natm):
                    for c in range(3):
                        gp = GOSTSASH(displaced(mol, i, c, h), make_options(al, f))
                        gm = GOSTSASH(displaced(mol, i, c, -h), make_options(al, f))
                        self.assertEqual(gp.n_gaussian, gm.n_gaussian)
                        num = (gp.pressure_au_list - gm.pressure_au_list) / (2 * h)
                        np.testing.assert_allclose(ana[i, c], num, rtol=0,
                                                   atol=1e-6 * np.abs(ana).max())


class TestGOSTSASHGradient(unittest.TestCase):
    """Full analytic nuclear gradient against finite differences (was test_grad.py)."""

    @staticmethod
    def energy(mol, opts):
        mf = gostsash_for_scf(scf.RHF(mol), GOSTSASH(mol, dict(opts)))
        mf.conv_tol = 1e-12
        e = mf.kernel()
        assert mf.converged
        return mf, e

    def check_gradient(self, direct, atom=NH3, atom_list=(0, 1, 2), f=4):
        mol = make_mol(atom)
        opts = make_options(list(atom_list), f, direct=direct)
        mf, _ = self.energy(mol, opts)
        ana = mf.Gradients().kernel()
        h = 1e-4
        num = np.zeros_like(ana)
        for i in range(mol.natm):
            for c in range(3):
                ep = self.energy(displaced(mol, i, c, h), opts)[1]
                em = self.energy(displaced(mol, i, c, -h), opts)[1]
                num[i, c] = (ep - em) / (2 * h)
        np.testing.assert_allclose(ana, num, rtol=0, atol=1e-7)

    def test_gradient_direct(self):
        self.check_gradient(direct=True)

    def test_gradient_cached(self):
        self.check_gradient(direct=False)

    def test_gradient_distorted_nh3_direct(self):
        # 2026-11-24: max |ana - num| = 2.6e-9 Eh/Bohr
        self.check_gradient(direct=True, atom=NH3_DISTORTED, atom_list=(0, 1, 2), f=2)

    def test_gradient_distorted_nh3_cached(self):
        # plane without the N atom; 2026-11-24: max |ana - num| = 2.9e-9 Eh/Bohr
        self.check_gradient(direct=False, atom=NH3_DISTORTED, atom_list=(1, 2, 3), f=8)


class TestGOSTSASHOrientation(unittest.TestCase):
    """The plane orientation must change the energy (was test_orient.py).
    With the old fsolve-determined f both orientations gave identical energies.
    With the plain (unweighted) mean the total load sum(p A) also depended on the
    orientation (+26 % ring plane, -3 % CH2 plane), which inflated the difference
    to ~39 kJ/mol; the area-weighted mean keeps the total load at P sum(A)."""

    def test_orientation_matters(self):
        mol = make_mol(CP, unit='Ang')
        energies = []
        for al in ([0, 1, 2], [4, 9, 10]):   # ring plane, perpendicular CH2 plane
            mf = gostsash_for_scf(scf.RHF(mol), GOSTSASH(mol, make_options(al, 4)))
            mf.conv_tol = 1e-10
            energies.append(mf.kernel())
            self.assertTrue(mf.converged)
        # 2026-11-24, area-weighted mean: difference 2.2e-4 Eh (~0.58 kJ/mol) at 110 points;
        # the sign flips with more points (-5.7 kJ/mol at 1202, -6.3 kJ/mol at 2702)
        self.assertGreater(abs(energies[0] - energies[1]), 1e-4)


class TestSmallAreaPoints(unittest.TestCase):
    """Surface points with area <= MIN_SURFACE_AREA are removed (distorted Cp)."""

    OPTS = dict(npoints=110, atom_list=[0, 8, 9], pressure_mpa=10000, f_exponent=4)

    def test_small_points_removed(self):
        g = GOSTSASH(make_mol(CP_DISTORTED), dict(self.OPTS))
        self.assertGreater(g.n_removed_points, 0)
        self.assertGreater(g.areas.min(), MIN_SURFACE_AREA)
        self.assertEqual(g.surface_dict['gslice_by_atom'][-1][1], g.n_gaussian)
        self.assertEqual(len(g._outer_surface_dict['area']), g.n_gaussian)

    def test_point_removal_continuity(self):
        """Removing near-zero-area points must leave the other pressures unchanged.
        With the plain mean, the number of points entered <x^f> and every p_t jumped."""
        mol = make_mol(CP_DISTORTED)
        with mock.patch('pyscf.solvent.gostsash.MIN_SURFACE_AREA', 0.0):
            g_all = GOSTSASH(mol, dict(self.OPTS))
        g = GOSTSASH(mol, dict(self.OPTS))
        kept = g_all.areas > MIN_SURFACE_AREA
        self.assertEqual(np.count_nonzero(kept), g.n_gaussian)
        np.testing.assert_allclose(g_all.pressure_au_list[kept], g.pressure_au_list,
                                   rtol=1e-8, atol=0)

    def test_gradient_atom10(self):
        """Atom 10 carries the 3.7e-15 Bohr^2 point. Without removing it, the analytic
        gradient is off by up to ~14 Eh/Bohr, scaled by the local p_t/P (plane [0,8,9]);
        with the plain mean, finite differences were also off by ~0.5 because the
        number of points entered M."""
        mol = make_mol(CP_DISTORTED)
        mf, _ = TestGOSTSASHGradient.energy(mol, self.OPTS)
        ana = mf.Gradients().kernel()
        self.assertLess(np.abs(ana.sum(axis=0)).max(), 1e-8)
        h = 1e-4
        for c in range(3):
            ep = TestGOSTSASHGradient.energy(displaced(mol, 10, c, h), self.OPTS)[1]
            em = TestGOSTSASHGradient.energy(displaced(mol, 10, c, -h), self.OPTS)[1]
            self.assertAlmostEqual(ana[10, c], (ep - em) / (2 * h), delta=1e-7)


if __name__ == '__main__':
    unittest.main()
