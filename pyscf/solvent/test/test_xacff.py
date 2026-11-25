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
Tests for X-ACFF with the anisotropic pressure distribution
p_t = P * |d_t|^f / M, M = sum(A |d|^f) / sum(A), d_t = distance to the plane.
'''

import unittest
import numpy as np
from pyscf import gto
from pyscf.solvent.xacff import xacff_external_terms
from pyscf.solvent.xhcff import xhcff_external_terms

# Bohr; randomly distorted NH3 and cyclopentadiene (sigma = 0.1 Angstrom, numpy seed 7),
# same geometries as in test_gostsash.py
NH3_DISTORTED = [
    ['N', [0.000232465294, 0.056454724683, 0.174962588247]],
    ['H', [-0.168297466402, 1.690422231010, -0.697620093764]],
    ['H', [1.542043654603, -0.634907302339, -0.603239605311]],
    ['H', [-1.647930923680, -0.795604546631, -0.442784183349]]]

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


def make_mol(atom):
    return gto.M(atom=atom, unit='Bohr', basis='sto-3g', verbose=0)


def options(atom_list, f, **kw):
    opts = dict(pressure_mpa=10_000.0, npoints=110, scaling_factor=1.2,
                rescale_forces=True, atom_list=list(atom_list), f_exponent=f)
    opts.update(kw)
    return opts


def fd_jacobian(atom, opts, h=1e-5):
    """Central finite differences of g_ext, shape (natm, natm, 3, 3) = [i, j, a, b]."""
    coords = make_mol(atom).atom_coords()
    natm = len(atom)
    jac = np.zeros((natm, natm, 3, 3))
    for j in range(natm):
        for b in range(3):
            g = []
            for s in (1, -1):
                xyz = coords.copy()
                xyz[j, b] += s * h
                mol = make_mol([[atom[k][0], xyz[k]] for k in range(natm)])
                g.append(xacff_external_terms(mol, **opts)[0])
            jac[:, j, :, b] = (g[0] - g[1]) / (2 * h)
    return jac


class TestXACFFOptions(unittest.TestCase):

    def test_f_exponent_required(self):
        opts = options([0, 1, 2], 4)
        del opts['f_exponent']
        with self.assertRaises(ValueError):
            xacff_external_terms(make_mol(NH3_DISTORTED), **opts)

    def test_f_exponent_below_one(self):
        with self.assertRaises(ValueError):
            xacff_external_terms(make_mol(NH3_DISTORTED), **options([0, 1, 2], 0.5))

    def test_f_exponent_not_a_number(self):
        with self.assertRaises(ValueError):
            xacff_external_terms(make_mol(NH3_DISTORTED), **options([0, 1, 2], '4'))

    def test_plane_width_removed(self):
        with self.assertRaises(ValueError):
            xacff_external_terms(make_mol(NH3_DISTORTED), **options([0, 1, 2], 4, plane_width=1.0))


class TestXACFFGradient(unittest.TestCase):

    def test_translational_invariance(self):
        g, h = xacff_external_terms(make_mol(CP_DISTORTED), **options([2, 5, 6], 4))
        self.assertLess(np.abs(g.sum(axis=0)).max(), 1e-12)
        self.assertLess(np.abs(h.sum(axis=0)).max(), 1e-10)

    def test_anisotropy(self):
        """The force differs from X-HCFF and depends on the plane orientation.
        With the old fsolve-determined f (~0) X-ACFF was practically X-HCFF."""
        mol = make_mol(CP_DISTORTED)
        g_ring = xacff_external_terms(mol, **options([0, 1, 2], 4))[0]
        g_ch2 = xacff_external_terms(mol, **options([4, 9, 10], 4))[0]
        g_hyd = xhcff_external_terms(mol, pressure_mpa=10_000.0, npoints=110,
                                     scaling_factor=1.2, rescale_forces=True)[0]
        self.assertGreater(np.abs(g_ring - g_hyd).max(), 1e-3)
        self.assertGreater(np.abs(g_ring - g_ch2).max(), 1e-3)


class TestXACFFHessian(unittest.TestCase):
    """Analytic Hessian (Jacobian of g_ext, incl. the derivative of p_t) against
    finite differences of the analytic gradient."""

    def check(self, atom, atom_list, f):
        opts = options(atom_list, f)
        mol = make_mol(atom)
        jac_ana = xacff_external_terms(mol, symmetrize=False, **opts)[1]
        jac_fd = fd_jacobian(atom, opts)
        scale = np.abs(jac_ana).max()
        np.testing.assert_allclose(jac_ana, jac_fd, rtol=0, atol=1e-6 * scale)
        h_sym = xacff_external_terms(mol, **opts)[1]
        np.testing.assert_allclose(
            h_sym, 0.5 * (jac_fd + jac_fd.transpose(1, 0, 3, 2)), rtol=0, atol=1e-6 * scale)

    def test_hessian_nh3_plane_with_n(self):
        self.check(NH3_DISTORTED, (0, 1, 2), 2)

    def test_hessian_nh3_plane_without_n(self):
        self.check(NH3_DISTORTED, (1, 2, 3), 8)

    def test_hessian_cp(self):
        self.check(CP_DISTORTED, (2, 5, 6), 4)

    def test_hessian_plane_groups_nh3(self):
        """Averaged and overlapping plane points (atom 2 in two groups)."""
        self.check(NH3_DISTORTED, ([1, 2], [2, 3], 0), 4)

    def test_hessian_plane_groups_cp(self):
        """CH2 midpoint as anchor point, ring C=C midpoint as third point."""
        self.check(CP_DISTORTED, ([9, 10], 4, [2, 3]), 4)

    def test_single_atom_groups_unchanged(self):
        mol = make_mol(CP_DISTORTED)
        g0, h0 = xacff_external_terms(mol, symmetrize=False, **options([2, 5, 6], 4))
        g1, h1 = xacff_external_terms(mol, symmetrize=False, **options([[2], [5], (6,)], 4))
        np.testing.assert_array_equal(g0, g1)
        np.testing.assert_array_equal(h0, h1)


if __name__ == '__main__':
    unittest.main()
