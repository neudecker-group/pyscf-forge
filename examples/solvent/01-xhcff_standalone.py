#!/usr/bin/env python
"""
Standalone X-HCFF prototype (no SCF wrapper class).

Adds X-HCFF external gradient and external analytic Hessian contributions
on top of regular PySCF SCF gradients/Hessians.
"""

import numpy as np
from pyscf import gto, scf
from pyscf.solvent.xhcff import xhcff_external_terms


def main():
    mol = gto.M(
        atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
        basis='cc-pVDZ',
        verbose=4,
    )

    pressure_mpa = 10_000.0
    scaling = 1.2
    npts = 302
    use_rescaling = True

    mf = scf.RHF(mol).run()

    g0 = mf.Gradients().kernel()
    h0 = mf.Hessian().kernel()

    g_ext, h_ext = xhcff_external_terms(
        mol,
        pressure_mpa=pressure_mpa,
        npoints=npts,
        scaling_factor=scaling,
        rescale_forces=use_rescaling,
    )

    g_tot = g0 + g_ext
    h_tot = h0 + h_ext

    np.set_printoptions(precision=8, suppress=True)
    print('\n=== X-HCFF standalone prototype ===')
    print(f'Pressure [MPa]: {pressure_mpa}')
    print(f'Scaling factor: {scaling}')
    print(f'N tess points:  {npts}')
    print(f'Rescaled (1/s^2): {use_rescaling}')
    print('\nExternal gradient contribution (Hartree/Bohr):')
    print(g_ext)
    print('\nTotal gradient (SCF + X-HCFF):')
    print(g_tot)
    print('\n||g_ext||_F =', np.linalg.norm(g_ext))
    print('||h_ext||_F =', np.linalg.norm(h_ext))
    print('||h_tot||_F =', np.linalg.norm(h_tot))


if __name__ == '__main__':
    main()
