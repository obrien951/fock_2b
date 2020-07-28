import time
import numpy as np
np.set_printoptions(precision=5, linewidth=200, suppress=True)
import psi4
import sys
import os

sys.path.append('/theoryfs2/ds/obrien/monika_f12/jk_test')
sys.path.append('/theoryfs2/ds/obrien/monika_f12/jk_test/basis')

# Memory for Psi4 in GB
psi4.set_memory('2 GB')
psi4.core.set_output_file("output.dat", False)

mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")

psi4.set_options({'basis':'aug-cc-pvdz',
                  'df_basis_mp2':'aug-cc-pvdz-ri',
                  'scf_type': 'df',
                  'e_convergence': 1e-8,
                  'd_convergence': 1e-8,
                })

ri_basis = 'aug-cc-pvdz-ri'


# Compute the reference wavefunction and CPHF using Psi
scf_e, scf_wfn = psi4.energy('SCF', return_wfn=True)
ndocc = scf_wfn.nalpha()
C1 = np.asarray(scf_wfn.Ca())
Co1 = C1[:, :ndocc]

mints = psi4.core.MintsHelper(scf_wfn.basisset())
conv1 = psi4.core.BasisSet.build(mol,'BASIS','aug-cc-pvdz')
conv2 = psi4.core.BasisSet.build(mol,'BASIS','aug-cc-pvtz')
aori_basis = psi4.core.BasisSet.build(mol,'BASIS',psi4.core.get_global_option('BASIS')+'_'+ri_basis)


I1 = mints.ao_eri(conv1,aori_basis,conv1,conv1)
I2 = mints.ao_eri(conv1,conv1,aori_basis,conv1)
# Build density matrix from occupied orbitals
D = np.einsum('pi,qi->pq', Co1, Co1)
# Build J and K matrices
J = np.einsum('pqrs,rs->pq', I1, D)
K = np.einsum('prqs,rs->pq', I2, D)



