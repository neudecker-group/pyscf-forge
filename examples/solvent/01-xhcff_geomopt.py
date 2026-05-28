#!/usr/bin/env python

'''
CO2 monomer geometry optimization and frequency analysis under hydrostatic
pressure using the X-HCFF model.

X-HCFF (eXternal Hydrostatic Compression Force Field) applies isotropic
pressure to a molecular cavity, compressing the molecule via an external
gradient contribution that is independent of the underlying QM method.

Reference:
    J. Chem. Phys. 153, 024105 (2020)
    https://doi.org/10.1063/5.0024671
'''

import numpy as np
from pyscf import gto, scf
from pyscf.hessian.thermo import harmonic_analysis
from pyscf.solvent.xhcff import xhcff_gradients_for_scf, xhcff_hessian_for_scf

# CO2 monomer, linear along z-axis
mol = gto.M(
    atom='C 0 0 0; O 0 0 1.16; O 0 0 -1.16',
    basis='cc-pVDZ',
    unit='Angstrom',
    verbose=4,
)

PRESSURE_MPA = 100_000.0   # 100 GPa
NPOINTS = 302
SCALING = 1.0

xhcff_opts = dict(pressure_mpa=PRESSURE_MPA, npoints=NPOINTS, scaling_factor=SCALING)


def co_bond_length(mol_opt):
    """C-O bond length (Bohr) between atom 1 (O) and atom 0 (C)."""
    return np.linalg.norm(mol_opt.atom_coord(1) - mol_opt.atom_coord(0))


#
# 1. Gas-phase geometry optimization (reference)
#
mf_gas = scf.RHF(mol).run()
grad_gas = mf_gas.Gradients().as_scanner()
mol_gas = grad_gas.optimizer().kernel()

#
# 2. Gas-phase frequency analysis
#
mf_gas_opt = scf.RHF(mol_gas).run()
hess_gas = mf_gas_opt.Hessian().kernel()
freq_gas = harmonic_analysis(mol_gas, hess_gas)

#
# 3. Geometry optimization under X-HCFF pressure
#
mf = scf.RHF(mol).run()
grad_press = xhcff_gradients_for_scf(mf, **xhcff_opts).as_scanner()
mol_press = grad_press.optimizer().kernel()

#
# 4. Frequency analysis at the compressed geometry
#
mf_opt = scf.RHF(mol_press).run()
hess_press = xhcff_hessian_for_scf(mf_opt, **xhcff_opts).kernel()
freq_press = harmonic_analysis(mol_press, hess_press)

#
# 5. Summary
#
r_gas   = co_bond_length(mol_gas)
r_press = co_bond_length(mol_press)
print(f'\nGas-phase   C-O bond: {r_gas:.4f} Bohr  ({r_gas * 0.529177:.4f} Angstrom)')
print(f'Compressed  C-O bond: {r_press:.4f} Bohr  ({r_press * 0.529177:.4f} Angstrom)')
print(f'Bond compression: {(r_gas - r_press):.4f} Bohr  ({(r_gas - r_press) * 0.529177:.4f} Angstrom)')
print(f'\nGas-phase vibrational frequencies (cm^-1):\n{freq_gas["freq_wavenumber"]}')
print(f'\nCompressed vibrational frequencies (cm^-1):\n{freq_press["freq_wavenumber"]}')
