#!/usr/bin/env python

'''
1-bromo-2-chloroethane (anti conformer): X-HCFF force norms as a function
of cavity scaling factor, with and without force rescaling.

Reproduces the right panel of Figure 3 from:
    J. Comput. Chem. (2025)
    https://doi.org/10.1002/jcc.70024

Key physics: Without the 1/s^2 rescaling, the XHCFF force norm grows
with the cavity scaling factor s because the cavity surface area grows
as ~s^2.  The 1/s^2 rescaling compensates exactly, making the force
norm approximately independent of s and giving a physically consistent
effective pressure regardless of cavity size.

Note: xhcff_external_terms depends only on molecular geometry, not on
the wavefunction, so no SCF calculation is required for this sweep.
'''

import numpy as np
from pyscf import gto
from pyscf.solvent.xhcff import xhcff_external_terms

ATOM = '''
C          0.02681        0.49079       -2.62962
C         -0.05658        0.73890       -1.13007
H         -0.24045       -0.56340       -2.85431
H          1.05878        0.68829       -2.98916
Cl        -1.09975        1.56714       -3.49300
Br         1.16840       -0.43138       -0.19047
H         -1.08817        0.54091       -0.76933
H          0.21129        1.79282       -0.90420
'''

mol = gto.M(atom=ATOM, basis='sto-3g', unit='Angstrom', verbose=0)

PRESSURE_MPA  = 10_000.0            # 10 GPa
NPOINTS       = 302
SCALE_FACTORS = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

norms_raw    = []
norms_rescaled = []

for s in SCALE_FACTORS:
    g_raw, _ = xhcff_external_terms(
        mol, pressure_mpa=PRESSURE_MPA, npoints=NPOINTS,
        scaling_factor=s, rescale_forces=False,
    )
    g_resc, _ = xhcff_external_terms(
        mol, pressure_mpa=PRESSURE_MPA, npoints=NPOINTS,
        scaling_factor=s, rescale_forces=True,
    )
    norms_raw.append(np.linalg.norm(g_raw))
    norms_rescaled.append(np.linalg.norm(g_resc))

print(f'\nX-HCFF gradient norms at {PRESSURE_MPA:.0f} MPa ({PRESSURE_MPA/1000:.0f} GPa)')
print(f'{"s":>6}  {"without rescaling":>18}  {"with rescaling":>15}')
print('-' * 44)
for s, nr, nrs in zip(SCALE_FACTORS, norms_raw, norms_rescaled):
    print(f'{s:>6.1f}  {nr:>18.5f}  {nrs:>15.5f}')

try:
    import matplotlib.pyplot as plt

    x = np.arange(len(SCALE_FACTORS))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x, norms_raw,      0.6,  label='Without rescaling', color='skyblue',    edgecolor='black', linewidth=0.8)
    ax.bar(x, norms_rescaled, 0.35, label='With rescaling',    color='lightcoral', edgecolor='black', linewidth=0.8)
    ax.set_xlabel('Scaling Factor')
    ax.set_ylabel('Norm of X-HCFF gradients / a.u.')
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in SCALE_FACTORS])
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('xhcff_rescaling.png', dpi=150)
    print('\nPlot saved to xhcff_rescaling.png')
except ImportError:
    pass
