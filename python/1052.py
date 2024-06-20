# # Unrestricted Open-Shell Hartree-Fock
# 
# In the first two tutorials in this module, we wrote programs which implement a closed-shell formulation of Hartree-Fock theory using restricted orbitals, aptly named Restricted Hartree-Fock (RHF).  In this tutorial, we will abandon strictly closed-shell systems and the notion of restricted orbitals, in favor of a more general theory known as Unrestricted Hartree-Fock (UHF) which can accommodate more diverse molecules.  In UHF, the orbitals occupied by spin up ($\alpha$) electrons and those occupied by spin down ($\beta$) electrons no longer have the same spatial component, e.g., 
# 
# $$\chi_i({\bf x}) = \begin{cases}\psi^{\alpha}_j({\bf r})\alpha(\omega) \\ \psi^{\beta}_j({\bf r})\beta(\omega)\end{cases},$$
# 
# meaning that they will not have the same orbital energy.  This relaxation of orbital constraints allows for more variational flexibility, which leads to UHF always being able to find a lower total energy solution than RHF.  
# 
# ## I. Theoretical Overview
# In UHF, we seek to solve the coupled equations
# 
# \begin{align}
# {\bf F}^{\alpha}{\bf C}^{\alpha} &= {\bf SC}^{\alpha}{\bf\epsilon}^{\alpha} \{\bf F}^{\beta}{\bf C}^{\beta} &= {\bf SC}^{\beta}{\bf\epsilon}^{\beta},
# \end{align}
# 
# which are the unrestricted generalizations of the restricted Roothan equations, called the Pople-Nesbitt equations.  Here, the one-electron Fock matrices are given by
# 
# \begin{align}
# F_{\mu\nu}^{\alpha} &= H_{\mu\nu} + (\mu\,\nu\mid\lambda\,\sigma)[D_{\lambda\sigma}^{\alpha} + D_{\lambda\sigma}^{\beta}] - (\mu\,\lambda\,\mid\nu\,\sigma)D_{\lambda\sigma}^{\beta}\F_{\mu\nu}^{\beta} &= H_{\mu\nu} + (\mu\,\nu\mid\,\lambda\,\sigma)[D_{\lambda\sigma}^{\alpha} + D_{\lambda\sigma}^{\beta}] - (\mu\,\lambda\,\mid\nu\,\sigma)D_{\lambda\sigma}^{\alpha},
# \end{align}
# 
# where the density matrices $D_{\lambda\sigma}^{\alpha}$ and $D_{\lambda\sigma}^{\beta}$ are given by
# 
# \begin{align}
# D_{\lambda\sigma}^{\alpha} &= C_{\sigma i}^{\alpha}C_{\lambda i}^{\alpha}\D_{\lambda\sigma}^{\beta} &= C_{\sigma i}^{\beta}C_{\lambda i}^{\beta}.
# \end{align}
# 
# Unlike for RHF, the orbital coefficient matrices ${\bf C}^{\alpha}$ and ${\bf C}^{\beta}$ are of dimension $M\times N^{\alpha}$ and $M\times N^{\beta}$, where $M$ is the number of AO basis functions and $N^{\alpha}$ ($N^{\beta}$) is the number of $\alpha$ ($\beta$) electrons.  The total UHF energy is given by
# 
# \begin{align}
# E^{\rm UHF}_{\rm total} &= E^{\rm UHF}_{\rm elec} + E^{\rm BO}_{\rm nuc},\;\;{\rm with}\E^{\rm UHF}_{\rm elec} &= \frac{1}{2}[({\bf D}^{\alpha} + {\bf D}^{\beta}){\bf H} + 
# {\bf D}^{\alpha}{\bf F}^{\alpha} + {\bf D}^{\beta}{\bf F}^{\beta}].
# \end{align}
# 
# ## II. Implementation
# 
# In any SCF program, there will be several common elements which can be abstracted from the program itself into separate modules, classes, or functions to 'clean up' the code that will need to be written explicitly; examples of this concept can be seen throughout the Psi4NumPy reference implementations.  For the purposes of this tutorial, we can achieve some degree of code cleanup without sacrificing readabilitiy and clarity by focusing on abstracting only the parts of the code which are both 
# - Lengthy subroutines, and 
# - Used repeatedly.  
# 
# In our UHF program, let's use what we've learned in the last tutorial by also implementing DIIS convergence accelleration for our SCF iterations.  With this in mind, two subroutines in particular would benefit from abstraction are
# 
# 1. Orthogonalize & diagonalize Fock matrix
# 2. Extrapolate previous trial vectors for new DIIS solution vector
# 
# Before we start writing our UHF program, let's try to write functions which can perform the above tasks so that we can use them in our implementation of UHF.  Recall that defining functions in Python has the following syntax:
# ~~~python
# def function_name(*args **kwargs):
#     # function block
#     return return_values
# ~~~
# A thorough discussion of defining functions in Python can be found [here](https://docs.python.org/2/tutorial/controlflow.html#defining-functions "Go to Python docs").  First, let's write a function which can diagonalize the Fock matrix and return the orbital coefficient matrix **C** and the density matrix **D**.  From our RHF tutorial, this subroutine is executed with:
# ~~~python
# F_p =  A.dot(F).dot(A)
# e, C_p = np.linalg.eigh(F_p)
# C = A.dot(C_p)
# C_occ = C[:, :ndocc]
# D = np.einsum('pi,qi->pq', C_occ, C_occ)
# ~~~
# Examining this code block, there are three quantities which must be specified beforehand:
# - Fock matrix, **F**
# - Orthogonalization matrix, ${\bf A} = {\bf S}^{-1/2}$
# - Number of doubly occupied orbitals, `ndocc`
# 
# However, since the orthogonalization matrix **A** is a static quantity (only built once, then left alone) we may choose to leave **A** as a *global* quantity, instead of an argument to our function.  In the cell below, using the code snippet given above, write a function `diag_F()` which takes **F** and the number of orbitals `norb` as arguments, and returns **C** and **D**:
# 

# ==> Define function to diagonalize F <==
def diag_F(F, norb):
    F_p = A.dot(F).dot(A)
    e, C_p = np.linalg.eigh(F_p)
    C = A.dot(C_p)
    C_occ = C[:, :norb]
    D = np.einsum('pi,qi->pq', C_occ, C_occ)
    return (C, D)


# Next, let's write a function to perform DIIS extrapolation and generate a new solution vector.  Recall that the DIIS-accellerated SCF algorithm is:
# #### Algorithm 1: DIIS within a generic SCF Iteration
# 1. Compute **F**, append to list of previous trial vectors
# 2. Compute AO orbital gradient **r**, append to list of previous residual vectors
# 3. Compute RHF energy
# 3. Check convergence criteria
#     - If RMSD of **r** sufficiently small, and
#     - If change in SCF energy sufficiently small, break
# 4. Build **B** matrix from previous AO gradient vectors
# 5. Solve Pulay equation for coefficients $\{c_i\}$
# 6. Compute DIIS solution vector **F_DIIS** from $\{c_i\}$ and previous trial vectors
# 7. Compute new orbital guess with **F_DIIS**
# 
# In our function, we will perform steps 4-6 of the above algorithm.  What information will we need to provide our function in order to do so?  To build **B** (step 4 above) in the previous tutorial, we used:
# ~~~python
# # Build B matrix
# B_dim = len(F_list) + 1
# B = np.empty((B_dim, B_dim))
# B[-1, :] = -1
# B[:, -1] = -1
# B[-1, -1] = 0
# for i in xrange(len(F_list)):
#     for j in xrange(len(F_list)):
#         B[i, j] = np.einsum('ij,ij->', DIIS_RESID[i], DIIS_RESID[j])
# ~~~
# Here, we see that we must have all previous DIIS residual vectors (`DIIS_RESID`), as well as knowledge about how many previous trial vectors there are (for the dimension of **B**).  To solve the Pulay equation (step 5 above):
# ~~~python
# # Build RHS of Pulay equation 
# rhs = np.zeros((B_dim))
# rhs[-1] = -1
#       
# # Solve Pulay equation for c_i's with NumPy
# coeff = np.linalg.solve(B, rhs)
# ~~~
# For this step, we only need the dimension of **B** (which we computed in step 4 above) and a NumPy routine, so this step doesn't require any additional arguments.  Finally, to build the DIIS Fock matrix (step 6):
# ~~~python
# # Build DIIS Fock matrix
# F = np.zeros_like(F_list[0])
# for x in xrange(coeff.shape[0] - 1):
#     F += coeff[x] * F_list[x]
# ~~~
# Clearly, for this step, we need to know all the previous trial vectors (`F_list`) and the coefficients we generated in the previous step.  In the cell below, write a funciton `diis_xtrap()` according to Algorithm 1 steps 4-6, using the above code snippets, which takes a list of previous trial vectors `F_list` and residual vectors `DIIS_RESID` as arguments and returns the new DIIS solution vector `F_DIIS`:
# 

# ==> Build DIIS Extrapolation Function <==
def diis_xtrap(F_list, DIIS_RESID):
    # Build B matrix
    B_dim = len(F_list) + 1
    B = np.empty((B_dim, B_dim))
    B[-1, :] = -1
    B[:, -1] = -1
    B[-1, -1] = 0
    for i in range(len(F_list)):
        for j in range(len(F_list)):
            B[i, j] = np.einsum('ij,ij->', DIIS_RESID[i], DIIS_RESID[j])

    # Build RHS of Pulay equation 
    rhs = np.zeros((B_dim))
    rhs[-1] = -1
      
    # Solve Pulay equation for c_i's with NumPy
    coeff = np.linalg.solve(B, rhs)
      
    # Build DIIS Fock matrix
    F_DIIS = np.zeros_like(F_list[0])
    for x in range(coeff.shape[0] - 1):
        F_DIIS += coeff[x] * F_list[x]
    
    return F_DIIS


# We are now ready to begin writing our UHF program!  Let's begin by importing <span style='font-variant: small-caps'> Psi4 </span> and NumPy, and defining our molecule & basic options:
# 

# ==> Import Psi4 & NumPy <==
import psi4
import numpy as np


# ==> Set Basic Psi4 Options <==
# Memory specification
psi4.set_memory(int(5e8))
numpy_memory = 2

# Set output file
psi4.core.set_output_file('output.dat', False)

# Define Physicist's water -- don't forget C1 symmetry!
mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")

# Set computation options
psi4.set_options({'guess': 'core',
                  'basis': 'cc-pvdz',
                  'scf_type': 'pk',
                  'e_convergence': 1e-8,
                  'reference': 'uhf'})


# You may notice that in the above `psi4.set_options()` block, there are two additional options -- namely, `'guess': 'core'` and `'reference': 'uhf'`.  These options make sure that when we ultimately check our program against <span style='font-variant: small-caps'> Psi4</span>, the options <span style='font-variant: small-caps'> Psi4 </span> uses are identical to our implementation.  Next, let's define the options for our UHF program; we can borrow these options from our RHF implementation with DIIS accelleration that we completed in our last tutorial.
# 

# ==> Set default program options <==
# Maximum SCF iterations
MAXITER = 40
# Energy convergence criterion
E_conv = 1.0e-6
D_conv = 1.0e-3


# Static quantities like the ERI tensor, core Hamiltonian, and orthogonalization matrix have exactly the same form in UHF as in RHF.  Unlike in RHF, however, we will need the number of $\alpha$ and $\beta$ electrons.  Fortunately, both these values are available through querying the Wavefunction object.  In the cell below, generate these static objects and compute each of the following:
# - Number of basis functions, `nbf`
# - Number of alpha electrons, `nalpha`
# - Number of beta electrons, `nbeta`
# - Number of doubly occupied orbitals, `ndocc` (Hint: In UHF, there can be unpaired electrons!)
# 

# ==> Compute static 1e- and 2e- quantities with Psi4 <==
# Class instantiation
wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('basis'))
mints = psi4.core.MintsHelper(wfn.basisset())

# Overlap matrix
S = np.asarray(mints.ao_overlap())

# Number of basis Functions, alpha & beta orbitals, and # doubly occupied orbitals
nbf = wfn.nso()
nalpha = wfn.nalpha()
nbeta = wfn.nbeta()
ndocc = min(nalpha, nbeta)

print('Number of basis functions: %d' % (nbf))
print('Number of singly occupied orbitals: %d' % (abs(nalpha - nbeta)))
print('Number of doubly occupied orbitals: %d' % (ndocc))

# Memory check for ERI tensor
I_size = (nbf**4) * 8.e-9
print('\nSize of the ERI tensor will be {:4.2f} GB.'.format(I_size))
memory_footprint = I_size * 1.5
if I_size > numpy_memory:
    psi4.core.clean()
    raise Exception("Estimated memory utilization (%4.2f GB) exceeds allotted memory                      limit of %4.2f GB." % (memory_footprint, numpy_memory))

# Build ERI Tensor
I = np.asarray(mints.ao_eri())

# Build core Hamiltonian
T = np.asarray(mints.ao_kinetic())
V = np.asarray(mints.ao_potential())
H = T + V

# Construct AO orthogonalization matrix A
A = mints.ao_overlap()
A.power(-0.5, 1.e-16)
A = np.asarray(A)


# Unlike the static quantities above, the CORE guess in UHF is slightly different than in RHF.  Since the $\alpha$ and $\beta$ electrons do not share spatial orbitals, we must construct a guess for *each* of the $\alpha$ and $\beta$ orbitals and densities.  In the cell below, using the function `diag_F()`, construct the CORE guesses and compute the nuclear repulsion energy:
# 
# (Hint: The number of $\alpha$ orbitals is the same as the number of $\alpha$ electrons!)
# 

# ==> Build alpha & beta CORE guess <==
Ca, Da = diag_F(H, nalpha)
Cb, Db = diag_F(H, nbeta)

# Get nuclear repulsion energy
E_nuc = mol.nuclear_repulsion_energy()


# We are almost ready to perform our SCF iterations; beforehand, however, we must initiate variables for the current & previous SCF energies, and the lists to hold previous residual vectors and trial vectors for the DIIS procedure.  Since, in UHF, there are Fock matrices ${\bf F}^{\alpha}$ and ${\bf F}^{\beta}$ for both $\alpha$ and $\beta$ orbitals, we must apply DIIS to each of these matrices separately.  In the cell below, define empty lists to hold previous Fock matrices and residual vectors for both $\alpha$ and $\beta$ orbitals:
# 

# ==> Pre-Iteration Setup <==
# SCF & Previous Energy
SCF_E = 0.0
E_old = 0.0


# We are now ready to write the SCF iterations.  The algorithm for UHF-SCF iteration, with DIIS convergence accelleration, is:
# #### Algorithm 2: DIIS within UHF-SCF Iteration
# 1. Build ${\bf F}^{\alpha}$ and ${\bf F}^{\beta}$, append to trial vector lists
# 2. Compute the DIIS residual for $\alpha$ and $\beta$, append to residual vector lists
# 3. Compute UHF energy
# 4. Convergence check
#     - If average of RMSD of $\alpha$ and $\beta$ residual sufficiently small, and
#     - If change in UHF energy sufficiently small, break
# 5. DIIS extrapolation of ${\bf F}^{\alpha}$ and ${\bf F}^{\beta}$ to form new solution vector
# 6. Compute new ${\alpha}$ and ${\beta}$ orbital & density guesses
# 
# In the cell below, write the UHF-SCF iteration according to Algorithm 2:
# 
# (Hint: Use your functions `diis_xtrap()` and `diag_F` for Algorithm 2 steps 5 & 6, respectively)
# 

# Trial & Residual Vector Lists -- one each for alpha & beta
F_list_a = []
F_list_b = []
R_list_a = []
R_list_b = []

# ==> UHF-SCF Iterations <==
print('==> Starting SCF Iterations <==\n')

# Begin Iterations
for scf_iter in range(MAXITER):
    # Build Fa & Fb matrices
    Ja = np.einsum('pqrs,rs->pq', I, Da)
    Jb = np.einsum('pqrs,rs->pq', I, Db)
    Ka = np.einsum('prqs,rs->pq', I, Da)
    Kb = np.einsum('prqs,rs->pq', I, Db)
    Fa = H + (Ja + Jb) - Ka
    Fb = H + (Ja + Jb) - Kb
    
    # Compute DIIS residual for Fa & Fb
    diis_r_a = A.dot(Fa.dot(Da).dot(S) - S.dot(Da).dot(Fa)).dot(A)
    diis_r_b = A.dot(Fb.dot(Db).dot(S) - S.dot(Db).dot(Fb)).dot(A)
    
    # Append trial & residual vectors to lists
    F_list_a.append(Fa)
    F_list_b.append(Fb)
    R_list_a.append(diis_r_a)
    R_list_b.append(diis_r_b)
    
    # Compute UHF Energy
    SCF_E = np.einsum('pq,pq->', (Da + Db), H)
    SCF_E += np.einsum('pq,pq->', Da, Fa)
    SCF_E += np.einsum('pq,pq->', Db, Fb)
    SCF_E *= 0.5
    SCF_E += E_nuc
    
    dE = SCF_E - E_old
    dRMS = 0.5 * (np.mean(diis_r_a**2)**0.5 + np.mean(diis_r_b**2)**0.5)
    print('SCF Iteration %3d: Energy = %4.16f dE = % 1.5E dRMS = %1.5E' % (scf_iter, SCF_E, dE, dRMS))
    
    # Convergence Check
    if (abs(dE) < E_conv) and (dRMS < D_conv):
        break
    E_old = SCF_E
    
    # DIIS Extrapolation
    if scf_iter >= 2:
        Fa = diis_xtrap(F_list_a, R_list_a)
        Fb = diis_xtrap(F_list_b, R_list_b)
    
    # Compute new orbital guess
    Ca, Da = diag_F(Fa, nalpha)
    Cb, Db = diag_F(Fb, nbeta)
    
    # MAXITER exceeded?
    if (scf_iter == MAXITER):
        psi4.core.clean()
        raise Exception("Maximum number of SCF iterations exceeded.")

# Post iterations
print('\nSCF converged.')
print('Final UHF Energy: %.8f [Eh]' % SCF_E)


# Congratulations! You've written your very own Unrestricted Hartree-Fock program with DIIS convergence accelleration!  Finally, let's check your final UHF energy against <span style='font-variant: small-caps'> Psi4</span>:
# 

# Compare to Psi4
SCF_E_psi = psi4.energy('SCF')
psi4.driver.p4util.compare_values(SCF_E_psi, SCF_E, 6, 'SCF Energy')


# ## References
# 1. A. Szabo and N. S. Ostlund, *Modern Quantum Chemistry*, Introduction to Advanced Electronic Structure Theory. Courier Corporation, 1996.
# 2. I. N. Levine, *Quantum Chemistry*. Prentice-Hall, New Jersey, 5th edition, 2000.
# 3. T. Helgaker, P. Jorgensen, and J. Olsen, *Molecular Electronic Structure Theory*, John Wiley & Sons Inc, 2000.
# 




# # Hartree-Fock Self-Consistent Field Theory
# ## I. Theoretical Overview
# In this tutorial, we will seek to introduce the theory and implementation of the quantum chemical method known as Hartree-Fock Self-Consistent Field Theory (HF-SCF) with restricted orbitals and closed-shell systems (RHF).  This theory seeks to solve the pseudo-eigenvalue matrix equation 
# 
# $$\sum_{\nu} F_{\mu\nu}C_{\nu i} = \epsilon_i\sum_{\nu}S_{\mu\nu}C_{\nu i}$$
# $${\bf FC} = {\bf SC\epsilon},$$
# 
# called the Roothan equations, which can be solved self-consistently for the orbital coefficient matrix **C** to and the orbital energy eigenvalues $\epsilon_i$.  The Fock matrix, **F**, has elements $F_{\mu\nu}$ given (in the atomic orbital basis) as
# 
# $$F_{\mu\nu} = H_{\mu\nu} + 2(\mu\,\nu\left|\,\lambda\,\sigma)D_{\lambda\sigma} - (\mu\,\lambda\,\right|\nu\,\sigma)D_{\lambda\sigma},$$
# 
# where $D_{\lambda\sigma}$ is an element of the one-particle density matrix **D**, constructed from the orbital coefficient matrix **C**:
# 
# $$D_{\lambda\sigma} = C_{\sigma i}C_{\lambda i}$$
# 
# Formally, the orbital coefficient matrix **C** is a $N\times M$ matrix, where $N$ is the number of atomic basis functions, and $M$ is the total number of molecular orbitals.  Physically, this matrix describes the contribution of every atomic basis function (columns) to a particular molecular orbital (e.g., the $i^{\rm th}$ row).  The density matrix **D** is a square matrix describing the electron density contained in each orbital.  In the molecular orbital basis, the density matrix has elements
# 
# $$D_{pq} = \left\{
# \begin{array}{ll}
# 2\delta_{pq} & p\; {\rm occupied} \0 & p\; {\rm virtual} \\end{array}\right .$$
# 
# The total RHF energy is given by
# 
# $$E^{\rm RHF}_{\rm total} = E^{\rm RHF}_{\rm elec} + E^{\rm BO}_{\rm nuc},$$
# 
# where $E^{\rm RHF}_{\rm elec}$ is the final electronic RHF energy, and $E^{\rm BO}_{\rm nuc}$ is the total nuclear repulsion energy within the Born-Oppenheimer approximation.  To compute the electronic energy, we may use the density matrix in the AO basis:
# 
# $$E^{\rm RHF}_{\rm elec} = (F_{\mu\nu} + H_{\mu\nu})D_{\mu\nu},$$
# 
# and the nuclear repulsion energy is simply
# 
# $$E^{\rm BO}_{\rm nuc} = \sum_{A>B}\frac{Z_AZ_B}{r_{AB}}$$
# 
# where $Z_A$ is the nuclear charge of atom $A$, and the sum runs over all unique nuclear pairs.
# 

# ## II. Implementation
# 
# Using the above overview, let's write a RHF program using <span style="font-variant: small-caps"> Psi4 </span> and NumPy.  First, we need to import these Python modules: 
# 

# ==> Import Psi4 & NumPy <==
import psi4
import numpy as np


# Next, using what you learned in the previous tutorial module, set the following <span style="font-variant: small-caps"> Psi4 </span> and molecule options.
# 
# Memory & Output specifications:
# - Give 500 Mb of memory to Psi4
# - Set Psi4 output file to "output.dat"
# - Set a variable `numpy_memory` to an acceptable amount of available memory for the working computer to use for storing tensors
# 
# Molecule definition:
# - Define the "physicist's water molecule" (O-H bond length = 1.1 Angstroms, HOH bond angle = 104 degrees)
# - Molecular symmetry C1
# 
# Computation options:
# - basis set cc-pVDZ
# - SCF type PK
# - Energy convergence criterion to 0.00000001
# 

# ==> Set Basic Psi4 Options <==
# Memory specification
psi4.set_memory(int(5e8))
numpy_memory = 2

# Set output file
psi4.core.set_output_file('output.dat', False)

# Define Physicist's water -- don't forget C1 symmetry!
mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")

# Set computation options
psi4.set_options({'basis': 'cc-pvdz',
                  'scf_type': 'pk',
                  'e_convergence': 1e-8})


# Since we will be writing our own, iterative RHF procedure, we will need to define options that we can use to tweak our convergence behavior.  For example, if something goes wrong and our SCF doesn't converge, we don't want to spiral into an infinite loop.  Instead, we can specify the maximum number of iterations allowed, and store this value in a variable called `maxiter`.  Here are some good default options for our program:
# ~~~python
# MAXITER = 40
# E_conv = 1.0e-6
# ~~~
# These are by no means the only possible values for these options, and it's encouraged to try different values and see for yourself how different choices affect the performance of our program.  For now, let's use the above as our default.
# 

# ==> Set default program options <==
# Maximum SCF iterations
MAXITER = 40
# Energy convergence criterion
E_conv = 1.0e-6


# Before we can build our Fock matrix, we'll need to compute the following static one- and two-electron quantities:
# 
# - Electron repulsion integrals (ERIs) **I** between our AOs
# - Overlap matrix **S**
# - Core Hamiltonian matrix **H**
# 
# Fortunately for us, we can do this using the machinery in <span style='font-variant: small-caps'> Psi4</span>.  In the first module, you learned about `psi4.core.Wavefunction` and `psi4.core.MintsHelper` classes.  In the cell below, use these classes to perform the following:
# 
# 1. Create Class Instances
# 
#     a. Build a wavefunction for our molecule and basis set
#     
#     b. Create an instance of the `MintsHelper` class with the basis set for the wavefunction
# 
# 2. Build overlap matrix, **S**
# 
#     a. Get the AO overlap matrix from `MintsHelper`, and cast it into a NumPy array
#     
#     b. Get the number of AO basis functions and number of doubly occupied orbitals from S and the wavefunciton
# 
# 3. Compute ERI Tensor, **I**
# 
#     a. Get ERI tensor from `MintsHelper`, and cast it into a NumPy array
# 
# 4. Build core Hamiltonian, **H**
# 
#     a. Get AO kinetic energy matrix from `MintsHelper`, and cast it into a NumPy array
# 
#     b. Get AO potential energy matrix from `MintsHelper`, and cast it into a NumPy array
# 
#     c. Build core Hamiltonian from kinetic & potential energy matrices
# 

# ==> Compute static 1e- and 2e- quantities with Psi4 <==
# Class instantiation
wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('basis'))
mints = psi4.core.MintsHelper(wfn.basisset())

# Overlap matrix
S = np.asarray(mints.ao_overlap())

# Number of basis Functions & doubly occupied orbitals
nbf = S.shape[0]
ndocc = wfn.nalpha()

print('Number of occupied orbitals: %3d' % (ndocc))
print('Number of basis functions: %3d' % (nbf))

# Memory check for ERI tensor
I_size = (nbf**4) * 8.e-9
print('\nSize of the ERI tensor will be {:4.2f} GB.'.format(I_size))
memory_footprint = I_size * 1.5
if I_size > numpy_memory:
    psi4.core.clean()
    raise Exception("Estimated memory utilization (%4.2f GB) exceeds allotted memory                      limit of %4.2f GB." % (memory_footprint, numpy_memory))

# Build ERI Tensor
I = np.asarray(mints.ao_eri())

# Build core Hamiltonian
T = np.asarray(mints.ao_kinetic())
V = np.asarray(mints.ao_potential())
H = T + V


# The Roothan equations
# 
# $${\bf FC} = {\bf SC\epsilon}$$
# 
# are only *pseudo*-eigenvalue equations due to the presence of the overlap matrix **S** on the right hand side of the equation.  Normally, the AO basis set will not be orthonormal, so the overlap matrix **S** will not be unity and therefore cannot be ignored.  Let's check to see whether our AO basis is orthonormal:
# 

# ==> Inspecting S for AO orthonormality <==
hope = np.allclose(S, np.eye(S.shape[0]))
print('\nDo we have any hope that our AO basis is orthonormal? %s!' % (hope))


# Just as we'd expected -- looks like we can't ignore the AO overlap matrix.  Therefore, the Fock matrix **F** cannot simply be diagonalized to solve for the orbital coefficient matrix **C**.  There is still hope, however!  We can overcome this issue by transforming the AO basis so that all of our basis functions are orthonormal.  In other words, we seek a matrix **A** such that the transformation 
# 
# $${\bf A}^{\dagger}{\bf SA} = {\bf 1}$$
# 
# One method of doing this is called *symmetric orthogonalization*, which lets ${\bf A} = {\bf S}^{-1/2}$.  Then, 
# 
# $${\bf A}^{\dagger}{\bf SA} = {\bf S}^{-1/2}{\bf SS}^{-1/2} = {\bf S}^{-1/2}{\bf S}^{1/2} = {\bf S}^0 = {\bf 1},$$
# 
# and we see that this choice for **A** does in fact yield an orthonormal AO basis.  In the cell below, construct this transformation matrix using <span style='font-variant: small-caps'> Psi4</span>'s built-in `Matrix` class member function `power()` just like the following:
# ~~~python
# A = mints.ao_overlap()
# A.power(-0.5, 1.e-16)
# A = np.asarray(A)
# ~~~
# 

# ==> Construct AO orthogonalization matrix A <==
A = mints.ao_overlap()
A.power(-0.5, 1.e-16)
A = np.asarray(A)

# Check orthonormality
S_p = A.dot(S).dot(A)
new_hope = np.allclose(S_p, np.eye(S.shape[0]))

if new_hope:
    print('There is a new hope for diagonalization!')
else:
    print("Whoops...something went wrong. Check that you've correctly built the transformation matrix.")


# The drawback of this scheme is that we would now have to either re-compute the ERI and core Hamiltonian tensors in the newly orthogonal AO basis, or transform them using our **A** matrix (both would be overly costly, especially transforming **I**).  On the other hand, substitute ${\bf C} = {\bf AC}'$ into the Roothan equations:
# 
# \begin{align}
# {\bf FAC'} &= {\bf SAC}'{\bf \epsilon}\{\bf A}^{\dagger}({\bf FAC}')&= {\bf A}^{\dagger}({\bf SAC}'){\bf \epsilon}\({\bf A}^{\dagger}{\bf FA}){\bf C}'&= ({\bf A}^{\dagger}{\bf SA}){\bf C}'{\bf \epsilon}\{\bf F}'{\bf C}' &= {\bf 1C}'{\bf \epsilon}\{\bf F}'{\bf C}' &= {\bf C}'{\bf \epsilon}\\end{align}
# 
# Clearly, we have arrived at a canonical eigenvalue equation.  This equation can be solved directly for the transformed orbital coefficient matrix ${\bf C}'$ by diagonalizing the transformed Fock matrix, ${\bf F}'$, before transforming ${\bf C}'$ back into the original AO basis with ${\bf C} = {\bf AC}'$.  
# 
# Before we can get down to the business of using the Fock matrix **F** to compute the RHF energy, we first need to compute the orbital coefficient **C** matrix.  But, before we compute the **C** matrix, we first need to build **F**.  Wait...hold on a second.  Which comes first, **C** or **F**?  Looking at the Roothan equations more closely, we see that that both sides depend on the **C** matrix, since **F** is a function of the orbitals:
# 
# 
# $${\bf F}({\bf C}){\bf C} = {\bf SC\epsilon}\,;\;\;F_{\mu\nu} = H_{\mu\nu} + 2(\mu\,\nu\mid\lambda\,\sigma)C_{\sigma i}C_{\lambda i} - (\mu\,\lambda\,\mid\nu\,\sigma)C_{\sigma i}C_{\lambda i}.$$
# 
# Therefore technically, *neither* **F** nor **C** can come first!  In order to proceed, we instead begin with a *guess* for the Fock matrix, from which we obtain a guess at the **C** matrix.  Without orbital coefficients (and therefore without electron densities), the most logical starting point for obtaining a guess at the Fock matrix is to begin with the only component of **F** that does *not* involve densities: the core Hamiltonian, **H**.  Below, using the NumPy `np.linalg.eigh()` function, obtain coefficient and density matrices using the core guess:
# 
# 1. Obtain ${\bf F}'$ by transforming the core Hamiltonian with the ${\bf A}$ matrix
# 2. Diagonalize the transformed Fock matrix for $\epsilon$ and ${\bf C}'$
# 3. Use doubly-occupied slice of coefficient matrix to build density matrix
# 

# ==> Compute C & D matrices with CORE guess <==
# Transformed Fock matrix
F_p = A.dot(H).dot(A)

# Diagonalize F_p for eigenvalues & eigenvectors with NumPy
e, C_p = np.linalg.eigh(F_p)

# Transform C_p back into AO basis
C = A.dot(C_p)

# Grab occupied orbitals
C_occ = C[:, :ndocc]

# Build density matrix from occupied orbitals
D = np.einsum('pi,qi->pq', C_occ, C_occ)


# The final quantity we need to compute before we can proceed with our implementation of the SCF procedure is the Born-Oppenheimer nuclear repulsion energy, $E^{\rm BO}_{\rm nuc}$.  We could use the expression given above in $\S$1, however we can also obtain this value directly from <span style='font-variant: small-caps'> Psi4</span>'s `Molecule` class.  In the cell below, compute the nuclear repulsion energy using either method. 
# 

# ==> Nuclear Repulsion Energy <==
E_nuc = mol.nuclear_repulsion_energy()


# Within each SCF iteration, we'll have to perform a number of tensor contractions when building the Fock matrix, computing the total RHF energy, and performing several transformations.  Since the computationl expense of this process is related to the number of unique indices, the most intensive step of computing the total electronic energy will be performing the four-index contractions corresponding to building Coulomb and Exchange matrices **J** and **K**, with elements
# 
# \begin{align}
# J[D_{\lambda\sigma}]_{\mu\nu} &= (\mu\,\nu\mid\lambda\,\sigma)D_{\lambda\sigma}\K[D_{\lambda\sigma}]_{\mu\nu} &= (\mu\,\lambda\mid\nu\,\sigma)D_{\lambda\sigma},
# \end{align}
# 
# when building the Fock matrix.  Fortunately, once **J** and **K** have been built, the Fock matrix may be computed as a simple matrix addition, instead of element-wise:
# 
# $$ {\bf F} = {\bf H} + 2{\bf J} - {\bf K}.$$
# 
# Formation of the **J** and **K** matrices will be the most expensive step of the RHF procedure, scaling with respect to the number of AOs as ${\cal O}(N^4)$.  Strategies for building these marices efficiently, as well as different methods for handling these tensor contractions, will be discussed in greater detail in tutorials 2c and 2d in this module, respectively. 
# 
# Let's now write our SCF iterations according to the following algorithm:
# 
# #### Algorithm 1: SCF Iteration
# for scf_iter less than MAXITER, do:
# 1. Build Fock matrix
#     - Build the Coulomb matrix **J** 
#     - Build the Exchange matrix **K** 
#     - Form the Fock matrix
# 2. RHF Energy
#     - Compute total RHF energy   
#     - If change in RHF energy less than E_conv, break    
#     - Save latest RHF energy as E_old
# 3. Compute new orbital guess
#     - Transform Fock matrix to orthonormal AO basis    
#     - Diagonalize ${\bf F}'$ for $\epsilon$ and ${\bf C}'$    
#     - Back transform ${\bf C}'$ to AO basis    
#     - Form **D** from occupied orbital slice of **C**
# 

# ==> SCF Iterations <==
# Pre-iteration energy declarations
SCF_E = 0.0
E_old = 0.0

print('==> Starting SCF Iterations <==\n')

# Begin Iterations
for scf_iter in range(1, MAXITER + 1):
    # Build Fock matrix
    J = np.einsum('pqrs,rs->pq', I, D)
    K = np.einsum('prqs,rs->pq', I, D)
    F = H + 2*J - K
    
    # Compute RHF energy
    SCF_E = np.einsum('pq,pq->', (H + F), D) + E_nuc
    print('SCF Iteration %3d: Energy = %4.16f dE = % 1.5E' % (scf_iter, SCF_E, SCF_E - E_old))
    
    # SCF Converged?
    if (abs(SCF_E - E_old) < E_conv):
        break
    E_old = SCF_E
    
    # Compute new orbital guess
    F_p =  A.dot(F).dot(A)
    e, C_p = np.linalg.eigh(F_p)
    C = A.dot(C_p)
    C_occ = C[:, :ndocc]
    D = np.einsum('pi,qi->pq', C_occ, C_occ)
    
    # MAXITER exceeded?
    if (scf_iter == MAXITER):
        psi4.core.clean()
        raise Exception("Maximum number of SCF iterations exceeded.")

# Post iterations
print('\nSCF converged.')
print('Final RHF Energy: %.8f [Eh]' % (SCF_E))


# Congratulations! You've written your very own Restricted Hartree-Fock program!  Finally, let's check your final RHF energy against <span style='font-variant: small-caps'> Psi4</span>:
# 

# Compare to Psi4
SCF_E_psi = psi4.energy('SCF')
psi4.driver.p4util.compare_values(SCF_E_psi, SCF_E, 6, 'SCF Energy')


# ## References
# 1. A. Szabo and N. S. Ostlund, *Modern Quantum Chemistry*, Introduction to Advanced Electronic Structure Theory. Courier Corporation, 1996.
# 2. I. N. Levine, *Quantum Chemistry*. Prentice-Hall, New Jersey, 5th edition, 2000.
# 3. T. Helgaker, P. Jorgensen, and J. Olsen, *Molecular Electronic Structure Theory*, John Wiley & Sons Inc, 2000.
# 




"""Tutorial: Symmetry-Adapted Perturbation Theory (SAPT0)"""

__author__    = ["Daniel G. A. Smith", "Konrad Patkowski"]
__credit__    = ["Daniel G. A. Smith", "Konrad Patkowski"]

__copyright__ = "(c) 2014-2017, The Psi4NumPy Developers"
__license__   = "BSD-3-Clause"
__date__      = "2017-06-24"


# # Symmetry-Adapted Perturbation Theory in Atomic Orbitals (SAPT0-AO)
# 
# In this tutorial, we revisit the SAPT0 calculation of interaction energy in the water dimer from the previous example. If you have not yet examined the introductory SAPT0 tutorial, this example might be very confusing.
# The new element here is that all exchange corrections $E^{(100)}_{\rm exch}$, $E^{(200)}_{\rm exch-ind,resp}$, and $E^{(200)}_{\rm exch-disp}$, as well as the electrostatic energy $E^{(100)}_{\rm elst}$, are computed in the atomic-orbital (AO) basis rather than the conventional molecular-orbital (MO) one. This should not affect the results at all: in fact, we expect exactly the same numerical answers as in the previous example (note that we are using exactly the same level of theory, including the $S^2$ approximation in all exchange corrections).
# Why recast the SAPT0 expressions from MOs into AOs? The reasons we give here explain why the AO-based formalism has the *potential* to be more computationally efficient. It does not mean that the AO approach as implemented in this example is already more efficient than the MO one - in general, it is not! (Not that it really matters as this example takes just a few seconds to run).
# 1. There is no need for computing and storing all different kinds of SAPT0 integrals in the MO basis. In this specific example, the only MO integrals needed are the ones required for the dispersion amplitudes $t^{rs}_{ab}$ and CPHF coefficients (relaxed induction amplitudes) $C^r_a$,$C^s_b$, that is, the quantities involving excitation energy denominators. Once formed, these amplitudes will be back-transformed into the AO basis.
# 2. Most of the tensor contractions involving two-electron integrals (naturally, these are the most expensive contractions present) can be formulated in the language of *generalized Coulomb and exchange (JK) operators* (see below). In fact, the entire $E^{(100)}_{\rm exch}$ and $E^{(200)}_{\rm exch-ind,resp}$ terms can be implemented in this way (but not $E^{(200)}_{\rm exch-disp}$). The developers of Psi4 have put a lot of effort into optimizing the computation of generalized JK matrices and, as a rule of thumb, if your Psi4 algorithm can utilize generalized JK matrices, it will be a very good idea to do so.
# 3. The AO algorithm can maximize the computational savings when density-fitted integrals and intermediates are employed (however, this specific example does not use any density fitting).
# 
# The original formulation of SAPT involves only MOs, and the AO formulation is not very well documented in the SAPT literature. Therefore, we will explicitly list all the formulas for the SAPT0 corrections computed below - some of these expressions are nowhere to be found in the published papers! A limited set of AO expressions for closed-shell SAPT0 is given in [Hesselmann:2005], and the analogous formulas for open-shell, UHF-based SAPT were first reported in [Hapka:2012]. A corrected and extended set of AO SAPT(UHF) expressions can be found in the supplementary material to [Gonthier:2016], however, even this reference does not have everything that we need (the exchange-dispersion energy is only formulated with density fitting). Therefore, please pay close attention to the expressions that need to be implemented.
# 

# A simple Psi 4 input script to compute SAPT interaction energies
# All exchange corrections are computed using AO algorithms
#
# Created by: Konrad Patkowski
# Helper routines by: Daniel G. A. Smith
# Date: 6/8/17
# License: GPL v3.0
#

import time
import numpy as np
from helper_SAPT import *
np.set_printoptions(precision=5, linewidth=200, threshold=2000, suppress=True)
import psi4

# Set Psi4 & NumPy Memory Options
psi4.set_memory('2 GB')
psi4.core.set_output_file('output.dat', False)

numpy_memory = 2

# Set molecule to dimer
dimer = psi4.geometry("""
O   -0.066999140   0.000000000   1.494354740
H    0.815734270   0.000000000   1.865866390
H    0.068855100   0.000000000   0.539142770
--
O    0.062547750   0.000000000  -1.422632080
H   -0.406965400  -0.760178410  -1.771744500
H   -0.406965400   0.760178410  -1.771744500
symmetry c1
""")

psi4.set_options({'basis': 'jun-cc-pVDZ',
                  'e_convergence': 1e-8,
                  'd_convergence': 1e-8})

sapt = helper_SAPT(dimer, memory=8, algorithm='AO')


# # 1. Preparation of the matrix elements
# 
# Note that, similar to the previous example, we have performed HF calculations for both monomers and initialized the pertinent integrals and intermediates via a call to `helper_SAPT`. This time, we set the additional optional parameter `algorithm` to `'AO'` (the default is `'MO'`) so that the generalized JK matrix calculations are set up and ready to go.
# 
# It is now time to prepare AO-based intermediates and introduce the pertinent notation. With the capital letters $K,L,\ldots$ denoting the AO basis (for simplicity, we assume that this basis is the same for monomers A and B), the matrix $C_{Ka}$ (`Ci`) denotes the SCF vectors for the occupied orbitals on A, and matrices `Cj`, `Cr`, and `Cs` collect the similar vectors for occupied orbitals on B, virtual orbitals on A, and virtual orbitals on B, respectively. The quantity $P^{\rm A}_{KL}=C_{Ka}C_{La}$ (denoted in the code by `Pi`) is one half of the monomer-A density matrix in the AO basis: similarly, $P^{\rm B}$ (`Pj`) is one half of the AO density matrix for monomer B. `S` denotes the overlap matrix in the AO basis, and `I` is the two-electron integral matrix $(KL|MN)$ (in the (11|22) index order) taken straight from `mints.ao_eri()`.
# The key intermediates in AO-based SAPT, the generalized Coulomb and exchange matrices, exhibit highly inconsistent notation in the SAPT literature. The definitions from [Hesselmann:2005] and [Hapka:2012] differ by a factor of 2: we will follow [Hapka:2012] and define these matrices as
# 
# \begin{equation}
# {J}[{X}]_{KL} = (KL|MN) {X}_{MN}
# \end{equation}
# 
# \begin{equation}
# {K}[{X}]_{KL} = (KM|NL) {X}_{MN}
# \end{equation}
# 
# An alternative definition is employed in [Gonthier:2016] and involves an explicit summation over occupied orbitals (therefore, we have to specify explicitly the monomer A or B). The generalized JK matrices defined in this way (for any two matrices $A$,$B$) will be denoted by $\bar{J}$ and $\bar{K}$, respectively:
# 
# \begin{equation}
# \bar{J}^{\rm A}[{A,B}]_{KL} = (KL|MN) {A}_{Ma}{B}_{Na} 
# \end{equation}
# 
# \begin{equation}
# \bar{K}^{\rm A}[{A,B}]_{KL} = (KM|NL) {A}_{Ma}{B}_{Na} 
# \end{equation}
# 
# and the monomer-B quantities $\bar{J}^{\rm B}[{A,B}]$ and $\bar{K}^{\rm B}[{A,B}]$ are defined with the summation over $a$ replaced by the one over $b$. The generalized JK matrices in either notation reduce to the ordinary JK matrices as follows:
# 
# \begin{equation}
# J^{\rm A}\equiv J[P^{\rm A}]\equiv \bar{J}^{\rm A}[Ci,Ci] 
# \end{equation}
# 
# and the same for $K$. Going back to the notation of [Hapka:2012], the intermediates needed for SAPT-AO are both standard JK matrices for both monomers `(Jii, Kii, Jjj, Kjj)` and generalized matrices $J[P^{\rm A}S P^{\rm B}]$, $K[P^{\rm A}S P^{\rm B}]$ `(Jij,Kij)`. Note that $J[P^{\rm B}S P^{\rm A}]=J[P^{\rm A}S P^{\rm B}]$ and $K[P^{\rm B}S P^{\rm A}]=K[P^{\rm A}S P^{\rm B}]^T$, so it is sufficient to generate the JK matrices for just one of these two operators.
# The last intermediates that we need are the matrices of the monomer electrostatic potential $\omega^{\rm X}=v^{\rm X}+2J^{\rm X}$ and of the complete monomer Fock operator $F^{\rm X}=v^{\rm X}+2J^{\rm X}-K^{\rm X}$ for X=A,B, where $v^{\rm X}$ is the matrix of the nuclear attraction operator. In the code below, these matrices are stored as `w_A, w_B, h_A, h_B`.
# 

# Build intermediates
int_timer = sapt_timer('intermediates')
Pi = np.dot(sapt.orbitals['a'], sapt.orbitals['a'].T)
Pj = np.dot(sapt.orbitals['b'], sapt.orbitals['b'].T)

S = sapt.S
num_el_A = (2 * sapt.ndocc_A)
num_el_B = (2 * sapt.ndocc_B)

Ci = sapt.orbitals['a']
Cj = sapt.orbitals['b']
Cr = sapt.orbitals['r']
Cs = sapt.orbitals['s']

I = np.asarray(sapt.mints.ao_eri())

Jii, Kii = sapt.compute_sapt_JK(Ci, Ci)
Jjj, Kjj = sapt.compute_sapt_JK(Cj, Cj)

Jij, Kij = sapt.compute_sapt_JK(Ci, Cj, tensor=sapt.chain_dot(Ci.T, S, Cj))

w_A = sapt.V_A + 2 * Jii
w_B = sapt.V_B + 2 * Jjj

h_A = sapt.V_A + 2 * Jii - Kii
h_B = sapt.V_B + 2 * Jjj - Kjj

int_timer.stop()



# # 2. Electrostatic energy
# 
# In the AO formalism, the SAPT0 electrostatic energy $E^{(100)}_{\rm elst}$ is given by
# 
# \begin{equation}
# E^{(100)}_{\rm elst}=4P^{\rm A}\cdot J^{\rm B} + 2P^{\rm A}\cdot v^{\rm B} + 2P^{\rm B}\cdot v^{\rm A} + V_{\rm nuc},
# \end{equation}
# 
# where $V_{\rm nuc}$ is the intermolecular nuclear repulsion energy. We define the dot-product notation for two matrices as follows,
# 
# \begin{equation}
# A\cdot B = A_{KL}B_{KL}
# \end{equation}
# 
# 

### Build electrostatics
elst_timer = sapt_timer('electrostatics')
two_el = 2 * np.vdot(Pi, Jjj)
att_a = np.vdot(sapt.V_A, Pj)
att_b = np.vdot(sapt.V_B, Pi)
rep = sapt.nuc_rep
elst_ijij = 2 * (two_el + att_a + att_b) + rep

Elst10 = elst_ijij
sapt_printer('Elst10', Elst10)
elst_timer.stop()
### End electrostatics



# # 3. First-order exchange energy
# 
# The SAPT0 first-order exchange energy $E^{(100)}_{\rm exch}$ within the $S^2$ approximation and the density matrix formalism [Moszynski:1994b] can be recast into AOs as follows,
# 
# \begin{align}
# E^{(100)}_{\rm exch}=&- 2P^{\rm B}\cdot K^{\rm A} -2 (P^{\rm A}SP^{\rm B})\cdot (F^{\rm A}+F^{\rm B})\\ & +2 (P^{\rm B}SP^{\rm A}SP^{\rm B})\cdot \omega^{\rm A}+2 (P^{\rm A}SP^{\rm B}SP^{\rm A})\cdot \omega^{\rm B}\\ &-2 (P^{\rm A}SP^{\rm B})\cdot K[P^{\rm A}SP^{\rm B}]
# \end{align}
# 
# In the implementation below, the `sapt.chain_dot()` call is nothing more than a multiplication of a chain of matrices - a series of `np.dot()` calls.
# 

### Start exchange
exch_timer = sapt_timer('exchange')
exch = 0
exch -= 2 * np.vdot(Pj, Kii)
exch -= 2 * np.vdot(sapt.chain_dot(Pi, S, Pj), (h_A + h_B))

exch += 2 * np.vdot(sapt.chain_dot(Pj, S, Pi, S, Pj), w_A)
exch += 2 * np.vdot(sapt.chain_dot(Pi, S, Pj, S, Pi), w_B)

exch -= 2 * np.vdot(sapt.chain_dot(Pi, S, Pj), Kij)

Exch100 = exch
sapt_printer('Exch10(S^2)', Exch100)
exch_timer.stop()
### End E100 (S^2) Exchange



# # 4. Dispersion energy
# 
# As the SAPT0 dispersion energy $E^{(200)}_{\rm disp}$ involves an energy denominator, it is calculated in the MO formalism in exactly the same way as in the previous example:
# 
# \begin{equation}
# E^{(200)}_{\rm disp}=4t^{rs}_{ab}v^{ab}_{rs}\;\;\;\;\;\; t^{rs}_{ab}=\frac{v_{ab}^{rs}}{\epsilon_a+\epsilon_b-\epsilon_r-\epsilon_s}
# \end{equation}
# 
# 

### Start E200 Disp
disp_timer = sapt_timer('dispersion')
v_abrs = sapt.v('abrs')
v_rsab = sapt.v('rsab')
e_rsab = 1/(-sapt.eps('r', dim=4) - sapt.eps('s', dim=3) + sapt.eps('a', dim=2) + sapt.eps('b'))

Disp200 = 4 * np.einsum('rsab,rsab,abrs->', e_rsab, v_rsab, v_abrs)
sapt_printer('Disp20', Disp200)
### End E200 Disp



# # 5. Exchange dispersion energy
# 
# In order to compute the SAPT0 exchange-dispersion energy $E^{(200)}_{\rm exch-disp}$ using AO-based quantities, the dispersion amplitude $t^{rs}_{ab}$ needs to be backtransformed into the AO basis. Note that the resulting AO amplitude has no index symmetry whatsoever.
# 
# \begin{equation}
# t_{KM}^{LN}=t_{ab}^{rs} C_{Ka} C_{Mb} C_{Lr} C_{Ns}
# \end{equation}
# 
# The SAPT0-AO exchange dispersion energy is now given by
# 
# \begin{align}
# E^{(200)}_{\rm exch-disp} = & t_{KM}^{LN} \left[-2 (KN|ML)-2 S_{KN} (F^{\rm A})_{ML}-2 S_{ML} (F^{\rm B})_{KN} \right. \\ &
# -4 (KL|MQ) (SP^{\rm A})_{NQ}+2 (ML|KQ) (SP^{\rm A})_{NQ}-4 (MN|KQ) (SP^{\rm B})_{LQ}+2 (KN|MQ) (SP^{\rm B})_{LQ} \\ &
# -4 (\omega^{\rm A})_{MN} (SP^{\rm B}S)_{KL}+2 S_{KN} (\omega^{\rm A} P^{\rm B}S)_{ML}+2 S_{ML} (\omega^{\rm A} P^{\rm B}S)_{NK} \\ &
# -4 (\omega^{\rm B})_{KL} (SP^{\rm A}S)_{MN}+2 S_{ML} (\omega^{\rm B} P^{\rm A}S)_{KN}+2 S_{KN} (\omega^{\rm B} P^{\rm A}S)_{LM} \\ &
# +4 (KQ|MN) (SP^{\rm B}SP^{\rm A})_{LQ}+4 (PL|MN) (SP^{\rm B}SP^{\rm A})_{KP}+4 (KL|MS) (SP^{\rm A}SP^{\rm B})_{NS}+4 (KL|RN) (SP^{\rm A}SP^{\rm B})_{MR} \\ &
# \left. -2 S_{KN} K[P^{\rm B}SP^{\rm A}]_{ML}-2 S_{ML} K[P^{\rm B}SP^{\rm A}]_{NK}-2 (MS|KQ) (SP^{\rm B})_{LS} (SP^{\rm A})_{NQ}-2 (NR|LP) (SP^{\rm B})_{KR} (SP^{\rm A})_{MP}\right] 
# \end{align}
# 
# 

### Start E200 Exchange-Dispersion

# Build t_rsab
t_rsab = np.einsum('rsab,rsab->rsab', v_rsab, e_rsab)

#backtransform t_rsab to the AO basis
t_lsab = np.einsum('rsab,rl->lsab', t_rsab, Cr.T)
t_lnab = np.einsum('lsab,sn->lnab', t_lsab, Cs.T)
t_lnkb = np.einsum('lnab,ak->lnkb', t_lnab, Ci.T)
t_lnkm = np.einsum('lnkb,bm->lnkm', t_lnkb, Cj.T)

ExchDisp20 = - 2 * np.einsum('lnkm,knml->', t_lnkm, I)

ExchDisp20 -= 2 * np.einsum('lnkm,ml,kn->', t_lnkm, h_A, S)
ExchDisp20 -= 2 * np.einsum('lnkm,ml,kn->', t_lnkm, S, h_B)

interm = 2 * np.einsum('klmq,nq->klmn', I, np.dot(S, Pi))
ExchDisp20 -= 2 * np.einsum('lnkm,klmn->', t_lnkm, interm)
ExchDisp20 += np.einsum('lnkm,mlkn->', t_lnkm, interm)

interm = 2 * np.einsum('klmq,nq->klmn', I, np.dot(S, Pj))
ExchDisp20 -= 2 * np.einsum('lnkm,mnkl->', t_lnkm, interm)
ExchDisp20 += np.einsum('lnkm,knml->', t_lnkm, interm)

ExchDisp20 -= 4 * np.einsum('lnkm,mn,kl->', t_lnkm, w_A, sapt.chain_dot(S, Pj, S))
ExchDisp20 += 2 * np.einsum('lnkm,kn,ml->', t_lnkm, S, sapt.chain_dot(w_A, Pj, S))
ExchDisp20 += 2 * np.einsum('lnkm,ml,nk->', t_lnkm, S, sapt.chain_dot(w_A, Pj, S))

ExchDisp20 -= 4 * np.einsum('lnkm,kl,mn->', t_lnkm, w_B, sapt.chain_dot(S, Pi, S))
ExchDisp20 += 2 * np.einsum('lnkm,ml,kn->', t_lnkm, S, sapt.chain_dot(w_B, Pi, S))
ExchDisp20 += 2 * np.einsum('lnkm,kn,lm->', t_lnkm, S, sapt.chain_dot(w_B, Pi, S))

spbspa = sapt.chain_dot(S, Pj, S, Pi)
spaspb = sapt.chain_dot(S, Pi, S, Pj)
interm = np.einsum('kqmn,lq->klmn', I, spbspa)
interm += np.einsum('plmn,kp->klmn', I, spbspa)
interm += np.einsum('klms,ns->klmn', I, spaspb)
interm += np.einsum('klrn,mr->klmn', I, spaspb)
ExchDisp20 += 4 * np.einsum('lnkm,klmn->', t_lnkm, interm)

ExchDisp20 -= 2 * np.einsum('lnkm,kn,ml->', t_lnkm, S, Kij.T)
ExchDisp20 -= 2 * np.einsum('lnkm,ml,nk->', t_lnkm, S, Kij.T)

spa = np.dot(S, Pi)
spb = np.dot(S, Pj)
interm = np.einsum('kpmq,nq->kpmn', I, spa)
interm = np.einsum('kpmn,lp->klmn', interm, spb)
ExchDisp20 -= 2 * np.einsum('lnkm,mlkn->', t_lnkm, interm)
ExchDisp20 -= 2 * np.einsum('lnkm,nklm->', t_lnkm, interm)

sapt_printer('Exch-Disp20', ExchDisp20)
disp_timer.stop()
### End E200 Exchange-Dispersion



# # 6. CPHF coefficients and induction energy
# 
# The monomer CPHF coefficients $C^a_r$, $C^b_s$ and the SAPT0 induction energy $E^{(200)}_{\rm ind,resp}$ are computed in the MO formalism, identical to the previous example.
# 
# \begin{equation}
# E^{(200)}_{\rm ind,resp}=4\tilde{v}^{rb}_{ab}C^a_r+4\tilde{v}^{as}_{ab}C^b_s
# \end{equation}
# 
# 

### Start E200 Induction and Exchange-Induction

# E200Induction and CPHF orbitals
ind_timer = sapt_timer('induction')

CPHF_ra, Ind20_ba = sapt.chf('B', ind=True)
sapt_printer('Ind20,r (A<-B)', Ind20_ba)

CPHF_sb, Ind20_ab = sapt.chf('A', ind=True)
sapt_printer('Ind20,r (A->B)', Ind20_ab)

Ind20r = Ind20_ba + Ind20_ab



# # 7. Exchange induction energy
# 
# We will present the AO formula for $E^{(200)}_{\rm exch-ind,resp}({\rm A}\leftarrow {\rm B})$; the corresponding formula for the B$\leftarrow$A counterpart is obtained by interchanging the quantities (density matrices, JK matrices, $\ldots$) pertaining to A with those of B. We first need to backtransform the CPHF coefficients to the AO basis:
# 
# \begin{equation}
# C^{\rm CPHF}_{KL}=C^{a}_{r} C_{Ka} C_{Lr} 
# \end{equation}
# 
# The formula for the A$\leftarrow$B part of $E^{(200)}_{\rm exch-ind,resp}$ now becomes
# 
# \begin{align}
# E^{(200)}_{\rm exch-ind,resp}({\rm A}\leftarrow {\rm B})=& C^{\rm CPHF}\cdot\left[ 
# -2 K^{\rm B}-2 SP^{\rm B}F^{\rm A}-2 F^{\rm B}P^{\rm B}S-4 J[P^{\rm B}SP^{\rm A}] +2 K[P^{\rm A}SP^{\rm B}] \right. \\ & 
# +2 \omega^{\rm B}P^{\rm A}SP^{\rm B}S+2 SP^{\rm B}SP^{\rm A}\omega^{\rm B}+2 SP^{\rm B}\omega^{\rm A}P^{\rm B}S \\ & 
# \left. +4 J[P^{\rm B}SP^{\rm A}SP^{\rm B}]-2 SP^{\rm B}K[P^{\rm B}SP^{\rm A}]-2 K[P^{\rm A}SP^{\rm B}]P^{\rm B}S\right] 
# \end{align}
# 
# Note that this correction requires one additional generalized Coulomb matrix, $J[P^{\rm B}SP^{\rm A}SP^{\rm B}]$. In the code below, this matrix is computed and stored in the variable `Jjij`.
# 

# Exchange-Induction

# A <- B
CPHFA_ao = sapt.chain_dot(Ci, CPHF_ra.T, Cr.T)
ExchInd20_ab = -2 * np.vdot(CPHFA_ao, Kjj)
ExchInd20_ab -= 2 * np.vdot(CPHFA_ao, sapt.chain_dot(S, Pj, h_A))
ExchInd20_ab -= 2 * np.vdot(CPHFA_ao, sapt.chain_dot(h_B, Pj, S))

ExchInd20_ab -= 4 * np.vdot(CPHFA_ao, Jij)
ExchInd20_ab += 2 * np.vdot(CPHFA_ao, Kij)

ExchInd20_ab += 2 * np.vdot(CPHFA_ao, sapt.chain_dot(w_B, Pi, S, Pj, S))
ExchInd20_ab += 2 * np.vdot(CPHFA_ao, sapt.chain_dot(S, Pj, S, Pi, w_B))
ExchInd20_ab += 2 * np.vdot(CPHFA_ao, sapt.chain_dot(S, Pj, w_A, Pj, S))

Jjij, Kjij = sapt.compute_sapt_JK(Cj, Cj, tensor=sapt.chain_dot(Cj.T, S, Pi, S, Cj))

ExchInd20_ab += 4 * np.vdot(CPHFA_ao, Jjij)
ExchInd20_ab -= 2 * np.vdot(CPHFA_ao, sapt.chain_dot(S, Pj, Kij.T))
ExchInd20_ab -= 2 * np.vdot(CPHFA_ao, sapt.chain_dot(Kij, Pj, S))

sapt_printer('Exch-Ind20,r (A<-B)', ExchInd20_ab)

# B <- A
CPHFB_ao = sapt.chain_dot(Cj, CPHF_sb.T, Cs.T)
ExchInd20_ba = -2 * np.vdot(CPHFB_ao, Kii)
ExchInd20_ba -= 2 * np.vdot(CPHFB_ao, sapt.chain_dot(S, Pi, h_B))
ExchInd20_ba -= 2 * np.vdot(CPHFB_ao, sapt.chain_dot(h_A, Pi, S))

ExchInd20_ba -= 4 * np.vdot(CPHFB_ao, Jij)
ExchInd20_ba += 2 * np.vdot(CPHFB_ao, Kij.T)

ExchInd20_ba += 2 * np.vdot(CPHFB_ao, sapt.chain_dot(w_A, Pj, S, Pi, S))
ExchInd20_ba += 2 * np.vdot(CPHFB_ao, sapt.chain_dot(S, Pi, S, Pj, w_A))
ExchInd20_ba += 2 * np.vdot(CPHFB_ao, sapt.chain_dot(S, Pi, w_B, Pi, S))

Jiji, Kiji = sapt.compute_sapt_JK(Ci, Ci, tensor=sapt.chain_dot(Ci.T, S, Pj, S, Ci))

ExchInd20_ba += 4 * np.vdot(CPHFB_ao, Jiji)
ExchInd20_ba -= 2 * np.vdot(CPHFB_ao, sapt.chain_dot(S, Pi, Kij))
ExchInd20_ba -= 2 * np.vdot(CPHFB_ao, sapt.chain_dot(Kij.T, Pi, S))

sapt_printer('Exch-Ind20,r (A->B)', ExchInd20_ba)
ExchInd20r = ExchInd20_ba + ExchInd20_ab

ind_timer.stop()
### End E200 Induction and Exchange-Induction



# # 8. Summary table
# 
# All the SAPT0-AO interaction energy contributions have been calculated. All that is left to do is to print out the contributions and the total energy, and to compare the results with the SAPT0 corrections calculated directly by Psi4.
# 

print('SAPT0 Results')
print('-' * 70)
sapt_printer('Exch10 (S^2)', Exch100)
sapt_printer('Elst10', Elst10)
sapt_printer('Disp20', Disp200)
sapt_printer('Exch-Disp20', ExchDisp20)
sapt_printer('Ind20,r', Ind20r)
sapt_printer('Exch-Ind20,r', ExchInd20r)

print('-' * 70)
sapt0 = Exch100 + Elst10 + Disp200 + ExchDisp20 + Ind20r + ExchInd20r
sapt_printer('Total SAPT0', sapt0)

# ==> Compare to Psi4 <==
psi4.set_options({'df_basis_sapt':'aug-cc-pvtz-ri'})
psi4.energy('sapt0')
Eelst = psi4.get_variable('SAPT ELST ENERGY')
Eexch = psi4.get_variable('SAPT EXCH10(S^2) ENERGY')
Eind  = psi4.get_variable('SAPT IND20,R ENERGY')
Eexind  = psi4.get_variable('SAPT EXCH-IND20,R ENERGY')
Edisp  = psi4.get_variable('SAPT DISP20 ENERGY')
Eexdisp  = psi4.get_variable('SAPT EXCH-DISP20 ENERGY')
psi4.driver.p4util.compare_values(Eelst, Elst10, 6, 'Elst100')
psi4.driver.p4util.compare_values(Eexch, Exch100, 6, 'Exch100(S^2)')
psi4.driver.p4util.compare_values(Edisp, Disp200, 6, 'Disp200')
psi4.driver.p4util.compare_values(Eexdisp, ExchDisp20, 6, 'Exch-Disp200')
psi4.driver.p4util.compare_values(Eind, Ind20r, 6, 'Ind200,r')
psi4.driver.p4util.compare_values(Eexind, ExchInd20r, 6, 'Exch-Ind200,r')



# ## References
# 
# 1. A paper that first formulated some SAPT0 and SAPT(DFT) corrections in AOs: "Density-functional theory-symmetry-adapted intermolecular perturbation theory with density fitting: A new efficient method to study intermolecular interaction energies"
# 	> [[Hesselmann:2005](http://aip.scitation.org/doi/abs/10.1063/1.1824898)] A. Heßelmann, G. Jansen, and M. Schütz, *J. Chem. Phys.* **122**, 014103 (2005)
# 2. Introduction of the UHF-based open-shell SAPT (mostly in AOs): "Symmetry-adapted perturbation theory based on unrestricted Kohn-Sham orbitals for high-spin open-shell van der Waals complexes"
# 	> [[Hapka:2012](http://aip.scitation.org/doi/10.1063/1.4758455)] M. Hapka, P. S. Żuchowski, M. M. Szczęśniak, and G. Chałasiński, *J. Chem. Phys.* **137**, 164104 (2012)
# 3. A new efficient implementation of SAPT(UHF) (and more AO formulas): "Density-fitted open-shell symmetry-adapted perturbation theory and application to π-stacking in benzene dimer cation and ionized DNA base pair steps"
# 	> [[Gonthier:2016](http://aip.scitation.org/doi/10.1063/1.4963385)] J. F. Gonthier and C. D. Sherrill, *J. Chem. Phys.* **145**, 134106  (2016)
# 4. The density-matrix formalism for SAPT exchange corrections employed in this work: "Many‐body theory of exchange effects in intermolecular interactions. Density matrix approach and applications to He–F$^−$, He–HF, H$_2$–HF, and Ar–H$_2$ dimers"
# 	> [[Moszynski:1994b](http://aip.scitation.org/doi/abs/10.1063/1.467225)] R. Moszynski, B. Jeziorski, S. Rybak, K. Szalewicz, and H. L. Williams, *J. Chem. Phys.* **100**, 5080 (1994)
# 




"""Tutorial: Symmetry-Adapted Perturbation Theory (SAPT0)"""

__author__    = ["Daniel G. A. Smith", "Konrad Patkowski"]
__credit__    = ["Daniel G. A. Smith", "Konrad Patkowski"]

__copyright__ = "(c) 2014-2017, The Psi4NumPy Developers"
__license__   = "BSD-3-Clause"
__date__      = "2017-06-24"


# # Symmetry-Adapted Perturbation Theory (SAPT0)
# 
# Symmetry-adapted perturbation theory (SAPT) is a perturbation theory aimed specifically at calculating the interaction energy between two molecules. Compared to the more conventional supermolecular approach where the interaction energy is computed as the difference between the electronic energy of the complex and the sum of electronic energies for the individual molecules (monomers), $E_{\rm int}=E_{\rm AB}-E_{\rm A}-E_{\rm B}$, SAPT obtains the interaction energy directly - no subtraction of similar terms is needed. Even more important, the result is obtained as a sum of separate corrections accounting for the electrostatic, induction, dispersion, and exchange contributions to interaction energy, so the SAPT decomposition facilitates the understanding and physical interpretation of results. 
# In the wavefunction-based variant presented here [Jeziorski:1994], SAPT is actually a triple perturbation theory. The zeroth-order Hamiltonian is the sum of the monomer Fock operators, $H_0=F_{\rm A}+F_{\rm B}$, and the perturbation correction $E^{(nkl)}$ corresponds to $n$th, $k$th, and $l$th order effects, respectively, of the intermolecular interaction operator $V$, the monomer-A Moller-Plesset fluctuation potential $W_{\rm A}=H_{\rm A}-F_{\rm A}$, and an analogous monomer-B potential $W_{\rm B}$. Thus, the SAPT correction $E^{(nkl)}$ is of the $n$th order in the *intermolecular interaction* and of the $(k+l)$th order in the *intramolecular correlation*.
# In this example, we will calculate the interaction energy between two molecules at the simplest, SAPT0 level of theory [Parker:2014]. In SAPT0, intramolecular correlation is neglected, and intermolecular interaction is included through second order. Specifically,
# 
# \begin{equation}
# E_{\rm int}^{\rm SAPT0}=E^{(100)}_{\rm elst}+E^{(100)}_{\rm exch}+E^{(200)}_{\rm ind,resp}+E^{(200)}_{\rm exch-ind,resp}+E^{(200)}_{\rm disp}+E^{(200)}_{\rm exch-disp}
# \end{equation}
# 
# In this equation, the consecutive corrections account for the electrostatic, first-order exchange, induction, exchange induction, dispersion, and exchange dispersion effects, respectively. The additional subscript ``resp'' denotes that these corrections are computed including the monomer relaxation (response) effects at the coupled-perturbed Hartree-Fock (CPHF) level of theory.
# Before we proceed to the computation of the individual SAPT0 corrections, let us make two comments on the specifics of the calculation of the exchange corrections. The exchange terms stem from the symmetry adaptation, specifically, from the presence of the $(N_{\rm A}+N_{\rm B})$-electron antisymmetrizer ${\cal A}$ that enforces the antisymmetry of the wavefunction upon an interchange of a pair of electrons between the monomers. Typically, the full operator ${\cal A}$ in SAPT is approximated as ${\cal A}=1+{\cal P}$, where the *single-exchange operator* ${\cal P}=\sum_{a\in {\rm A}}\sum_{b\in {\rm B}}P_{ab}$ collects all transpositions of a single pair of electrons between the interacting molecules. This approach is known as the *single exchange approximation* or the *$S^2$ approximation* --- the latter name refers to keeping terms that are quadratic in the intermolecular overlap integrals $S$ and neglecting terms that vanish like $S^4$, $S^6$, $\ldots$. In Psi4,the $E^{(100)}_{\rm exch}$ correction can be computed without the $S^2$ approximation, and the nonapproximated formulas for $E^{(200)}_{\rm exch-ind,resp}$ and $E^{(200)}_{\rm exch-disp}$ have also been derived [Schaffer:2013]. Nevertheless, in this example we will employ the $S^2$ approximation in all exchange corrections. Second, there exist two formalisms for the derivation of SAPT exchange corrections: the second-quantization approach [Moszynski:1994a] and the density matrix formalism [Moszynski:1994b]. The two methodologies lead to completely different SAPT expressions which, however, lead to identical results as long as the full dimer basis set is employed. Below, we will adopt the density formalism that is more general (valid in dimer and monomer basis sets) and exhibits more favorable computational scaling (however, more different types of two-electron integrals are required).
# 

# # 1. Preparation of the matrix elements
# 
# The expressions for SAPT0 corrections contain similar quantities as the ones for other correlated electronic structure theories: one- and two-electron integrals over molecular orbitals (MOs), Hartree-Fock (HF) orbital energies, and various amplitudes and intermediates. The feature unique to SAPT is that one has two sets of occupied and virtual (unoccupied) MOs, one for molecule A and one for molecule B (the MOs for the two molecules are not mutually orthogonal, and they may span the same one-electron space but do not have to do so). The most direct consequence of having two sets of MOs is a large number of different MO-basis two-electron integrals $(xy\mid zw)$: each of the four indices can be an occupied orbital of A, a virtual orbital of A, an occupied orbital of B, or a virtual orbital of B. Even when we account for all possible index symmetries, a few dozen types of MO integrals are possible, and we need a code for the integral transformation from atomic orbitals (AOs) to MOs that can produce all of these types. This transformation, and a number of other useful routines, is present in the `helper_SAPT` module that one has to load at the beginning of the SAPT run.
# 

# A simple Psi 4 input script to compute SAPT interaction energies
#
# Created by: Daniel G. A. Smith
# Date: 12/1/14
# License: GPL v3.0
#

import time
import numpy as np
from helper_SAPT import *
np.set_printoptions(precision=5, linewidth=200, threshold=2000, suppress=True)
import psi4

# Set Psi4 & NumPy Memory Options
psi4.set_memory('2 GB')
psi4.core.set_output_file('output.dat', False)

numpy_memory = 2



# Next, we specify the geometry of the complex (in this example, it will be the water dimer). Note that we have to let Psi4 know which atoms belong to molecule A and which ones are molecule B. We then call the `helper_SAPT` function to initialize all quantities that will be needed for the SAPT corrections. In particular, the HF calculations will be performed for molecules A and B separately, and the two sets of orbital energies and MO coefficients will be waiting for SAPT to peruse.
# 

# Set molecule to dimer
dimer = psi4.geometry("""
O   -0.066999140   0.000000000   1.494354740
H    0.815734270   0.000000000   1.865866390
H    0.068855100   0.000000000   0.539142770
--
O    0.062547750   0.000000000  -1.422632080
H   -0.406965400  -0.760178410  -1.771744500
H   -0.406965400   0.760178410  -1.771744500
symmetry c1
""")

psi4.set_options({'basis': 'jun-cc-pVDZ',
                  'e_convergence': 1e-8,
                  'd_convergence': 1e-8})

sapt = helper_SAPT(dimer, memory=8)


# Before we start computing the SAPT0 corrections, we still need to specify the pertinent notation and define the matrix elements that we will be requesting from `helper_SAPT`. In the classic SAPT papers [Rybak:1991], orbital indices $a,a',a'',\ldots$ and $b,b',b'',\ldots$ denote occupied orbitals of monomers A and B, respectively. The virtual orbitals of monomers A and B are denoted by $r,r',r'',\ldots$ and $s,s',s'',\ldots$, respectively. The overlap integral $S^x_y=\langle x|\rangle y$ reduces to a Kronecker delta when two orbitals from the same monomer are involved, for example, $S^a_{a'}=\delta_{aa'}$, $S^a_r=0$, however, the intermolecular overlap integrals cannot be simplified in any general fashion. Any kind of overlap integral can be requested by calling `sapt.s`, for example, `sapt.s('ab')` gives the $S^a_b$ matrix. For the convenience of implementation, the one-electron (nuclear attraction) $(v_{\rm X})^x_y$ (X = A or B) and nuclear repulsion $V_0$ contributions are usually folded into the two-electron integrals $v^{xy}_{zw}\equiv (xz|yw)$ forming the *dressed* integrals $\tilde{v}$:
# 
# \begin{equation}
# \tilde{v}^{xy}_{zw}=v^{xy}_{zw}+(v_{\rm A})^{y}_{w}S^{x}_{z}/N_{\rm A}+(v_{\rm B})^{x}_{z}S^{y}_{w}/N_{\rm B}+V_0S^{x}_{z}S^{y}_{w}/N_{\rm A}N_{\rm B},
# \end{equation}
# 
# where $N_{\rm X}$, X=A,B, is the number of electrons in monomer X. An arbitrary *dressed* integral $\tilde{v}^{xy}_{zw}$ can be requested by calling `sapt.vt('xyzw')`. Finally, the HF orbital energy for either monomer can be obtained by calling `sapt.eps`; for example, `sapt.eps('r')` returns a 1D array of virtual orbital energies for monomer A.
# 

# # 2. Electrostatic energy
# 
# The SAPT0 electrostatic energy $E^{(100)}_{\rm elst}$ is simply the expectation value of the intermolecular interaction operator $V$ over the zeroth-order wavefunction which is the product of HF determinants for monomers A and B. For the interaction of two closed-shell systems, this energy is obtained by a simple summation of *dressed* two-electron integrals over occupied orbitals of A and B:
# 
# \begin{equation}
# E^{(100)}_{\rm elst}=4\tilde{v}^{ab}_{ab}.
# \end{equation}
# 
# 


### Start E100 Electrostatics
elst_timer = sapt_timer('electrostatics')
Elst10 = 4 * np.einsum('abab', sapt.vt('abab'))
elst_timer.stop()
### End E100 Electrostatics



# # 3. First-order exchange energy
# 
# The SAPT0 first-order exchange energy $E^{(100)}_{\rm exch}$ within the $S^2$ approximation and the density matrix formalism is given by Eq. (40) of [Moszynski:1994b]:
# 
# \begin{align}
# E^{(100)}_{\rm exch}=&-2\left[\tilde{v}^{ba}_{ab}+S^b_{a'}\left(2\tilde{v}^{aa'}_{ab}-\tilde{v}^{a'a}_{ab}\right)+S^{a}_{b'}\left(2\tilde{v}^{b'b}_{ab}-\tilde{v}^{bb'}_{ab}\right)\right.\\ &\left.-2S^b_{a'}S^{a'}_{b'}\tilde{v}^{ab'}_{ab}-2S^{b'}_{a'}S^{a}_{b'}\tilde{v}^{a'b}_{ab}+S^b_{a'}S^{a}_{b'}\tilde{v}^{a'b'}_{ab}\right]
# \end{align}
# 
# and involves several different types of *dressed* MO integrals as well as some intermolecular overlap integrals (not that all indices still pertain to occupied orbitals in this formalism). In Psi4NumPy, each tensor contraction in the above expression can be performed with a single `np.einsum` call:
# 

### Start E100 Exchange
exch_timer = sapt_timer('exchange')
vt_abba = sapt.vt('abba')
vt_abaa = sapt.vt('abaa')
vt_abbb = sapt.vt('abbb')
vt_abab = sapt.vt('abab')
s_ab = sapt.s('ab')

Exch100 = np.einsum('abba', vt_abba)

tmp = 2 * vt_abaa - vt_abaa.swapaxes(2, 3)
Exch100 += np.einsum('Ab,abaA', s_ab, tmp)

tmp = 2 * vt_abbb - vt_abbb.swapaxes(2, 3)
Exch100 += np.einsum('Ba,abBb', s_ab.T, tmp)

Exch100 -= 2 * np.einsum('Ab,BA,abaB', s_ab, s_ab.T, vt_abab)
Exch100 -= 2 * np.einsum('AB,Ba,abAb', s_ab, s_ab.T, vt_abab)
Exch100 += np.einsum('Ab,Ba,abAB', s_ab, s_ab.T, vt_abab)

Exch100 *= -2
exch_timer.stop()
### End E100 (S^2) Exchange



# # 4. Dispersion energy
# 
# The SAPT0 dispersion energy $E^{(200)}_{\rm disp}$ is given by the formula
# 
# \begin{equation}
# E^{(200)}_{\rm disp}=4t^{rs}_{ab}v^{ab}_{rs}
# \end{equation}
# 
# where the *dispersion amplitude* $t^{rs}_{ab}$, representing a single excitation on A and a single excitation on B, involves a two-electron integral and an excitation energy denominator:
# 
# \begin{equation}
# t^{rs}_{ab}=\frac{v_{ab}^{rs}}{\epsilon_a+\epsilon_b-\epsilon_r-\epsilon_s}
# \end{equation}
# 
# Note that for this particular type of integral $\tilde{v}^{ab}_{rs}=v^{ab}_{rs}$: therefore, `sapt.v` instead of `sapt.vt` is used to prepare this tensor.
# 

### Start E200 Disp
disp_timer = sapt_timer('dispersion')
v_abrs = sapt.v('abrs')
v_rsab = sapt.v('rsab')
e_rsab = 1/(-sapt.eps('r', dim=4) - sapt.eps('s', dim=3) + sapt.eps('a', dim=2) + sapt.eps('b'))

Disp200 = 4 * np.einsum('rsab,rsab,abrs->', e_rsab, v_rsab, v_abrs)
### End E200 Disp



# # 5. Exchange dispersion energy
# 
# Some of the formulas for the SAPT0 exchange-dispersion energy $E^{(200)}_{\rm exch-disp}$ in the original papers contained errors. The corrected formula for this term is given by e.g. Eq. (10) of [Patkowski:2007]:
# 
# \begin{align}
# E^{(200)}_{\rm exch-disp}=&-2t^{ab}_{rs}\left[\tilde{v}^{sr}_{ab}+S^s_a (2\tilde{v}^{a'r}_{a'b}-\tilde{v}^{ra'}_{a'b})+ S^s_{a'} (2\tilde{v}^{ra'}_{ab}-\tilde{v}^{a'r}_{ab})\right.\\ &+ S^r_b (2\tilde{v}^{sb'}_{ab'}-\tilde{v}^{b's}_{ab'})+ S^r_{b'} (2\tilde{v}^{b's}_{ab}-\tilde{v}^{sb'}_{ab}) \\ &+S^{r}_{b} S^{b'}_{a'} \tilde{v}^{a's}_{ab'}-2 S^{r}_{b'} S^{b'}_{a'} \tilde{v}^{a's}_{ab}-2 S^{r}_{b} S^{b'}_{a} \tilde{v}^{a's}_{a'b'}+4 S^{r}_{b'} S^{b'}_{a} \tilde{v}^{a's}_{a'b} \\ &-2 S^{s}_{a} S^{a'}_{b} \tilde{v}^{rb'}_{a'b'}+4 S^{s}_{a'} S^{a'}_{b} \tilde{v}^{rb'}_{ab'}+ S^{s}_{a} S^{a'}_{b'} \tilde{v}^{rb'}_{a'b}-2 S^{s}_{a'} S^{a'}_{b'} \tilde{v}^{rb'}_{ab} \\ &+ S^{r}_{b'} S^{s}_{a'} \tilde{v}^{a'b'}_{ab}-2 S^{r}_{b} S^{s}_{a'} \tilde{v}^{a'b'}_{ab'}-2 S^{r}_{b'} S^{s}_{a} \tilde{v}^{a'b'}_{a'b} \\ &\left. + S^{a'}_{b} S^{b'}_{a} \tilde{v}^{rs}_{a'b'}-2 S^{a'}_{b} S^{b'}_{a'} \tilde{v}^{rs}_{ab'}-2 S^{a'}_{b'} S^{b'}_{a} \tilde{v}^{rs}_{a'b}\right]
# \end{align}
# 
# The corresponding Psi4NumPy code first recreates the dispersion amplitudes $t^{rs}_{ab}$ and then prepares the tensor `xd_absr` that is equal to the entire expression in brackets. The additional two intermediates `h_abrs` and `q_abrs` collect terms involving one and two overlap integrals, respectively.
# 

### Start E200 Exchange-Dispersion

# Build t_rsab
t_rsab = np.einsum('rsab,rsab->rsab', v_rsab, e_rsab)

# Build h_abrs
vt_abar = sapt.vt('abar')
vt_abra = sapt.vt('abra')
vt_absb = sapt.vt('absb')
vt_abbs = sapt.vt('abbs')

tmp = 2 * vt_abar - vt_abra.swapaxes(2, 3)
h_abrs = np.einsum('as,AbAr->abrs', sapt.s('as'), tmp)

tmp = 2 * vt_abra - vt_abar.swapaxes(2, 3)
h_abrs += np.einsum('As,abrA->abrs', sapt.s('as'), tmp)

tmp = 2 * vt_absb - vt_abbs.swapaxes(2, 3)
h_abrs += np.einsum('br,aBsB->abrs', sapt.s('br'), tmp)

tmp = 2 * vt_abbs - vt_absb.swapaxes(2, 3)
h_abrs += np.einsum('Br,abBs->abrs', sapt.s('br'), tmp)

# Build q_abrs
vt_abas = sapt.vt('abas')
q_abrs =      np.einsum('br,AB,aBAs->abrs', sapt.s('br'), sapt.s('ab'), vt_abas)
q_abrs -= 2 * np.einsum('Br,AB,abAs->abrs', sapt.s('br'), sapt.s('ab'), vt_abas)
q_abrs -= 2 * np.einsum('br,aB,ABAs->abrs', sapt.s('br'), sapt.s('ab'), vt_abas)
q_abrs += 4 * np.einsum('Br,aB,AbAs->abrs', sapt.s('br'), sapt.s('ab'), vt_abas)

vt_abrb = sapt.vt('abrb')
q_abrs -= 2 * np.einsum('as,bA,ABrB->abrs', sapt.s('as'), sapt.s('ba'), vt_abrb)
q_abrs += 4 * np.einsum('As,bA,aBrB->abrs', sapt.s('as'), sapt.s('ba'), vt_abrb)
q_abrs +=     np.einsum('as,BA,AbrB->abrs', sapt.s('as'), sapt.s('ba'), vt_abrb)
q_abrs -= 2 * np.einsum('As,BA,abrB->abrs', sapt.s('as'), sapt.s('ba'), vt_abrb)

vt_abab = sapt.vt('abab')
q_abrs +=     np.einsum('Br,As,abAB->abrs', sapt.s('br'), sapt.s('as'), vt_abab)
q_abrs -= 2 * np.einsum('br,As,aBAB->abrs', sapt.s('br'), sapt.s('as'), vt_abab)
q_abrs -= 2 * np.einsum('Br,as,AbAB->abrs', sapt.s('br'), sapt.s('as'), vt_abab)

vt_abrs = sapt.vt('abrs')
q_abrs +=     np.einsum('bA,aB,ABrs->abrs', sapt.s('ba'), sapt.s('ab'), vt_abrs)
q_abrs -= 2 * np.einsum('bA,AB,aBrs->abrs', sapt.s('ba'), sapt.s('ab'), vt_abrs)
q_abrs -= 2 * np.einsum('BA,aB,Abrs->abrs', sapt.s('ba'), sapt.s('ab'), vt_abrs)

# Sum it all together
xd_absr = sapt.vt('absr')
xd_absr += h_abrs.swapaxes(2, 3)
xd_absr += q_abrs.swapaxes(2, 3)
ExchDisp20 = -2 * np.einsum('absr,rsab->', xd_absr, t_rsab)

disp_timer.stop()
### End E200 Exchange-Dispersion



# # 6. CPHF coefficients and induction energy
# 
# As already mentioned, the induction and exchange-induction contributions to SAPT0 are calculated including the relaxation of one molecule's HF orbitals in the electrostatic potential generated by the other molecule. Mathematically, this relaxation is taken into account by computing the CPHF coefficients $C^a_r$ for monomer A [Caves:1969] that specify the linear response of the HF orbitals of A to the electrostatic potential $\omega_{\rm B}$ generated by the nuclei and electrons of the (unperturbed) monomer B and the analogous coefficients $C^b_s$ that describe the response of B to the electrostatic potential of A. The CPHF coefficients are computed by solving the system of equations
# 
# \begin{equation}
# (\epsilon_r-\epsilon_a)C^a_r+(2v^{ar'}_{ra'}-v^{r'a}_{ra'})C^{a'}_{r'}+(2v^{aa'}_{rr'}-v^{a'a}_{rr'})C^{r'}_{a'}=-2\tilde{v}^{ab}_{rb}. 
# \end{equation}
# 
# and similarly for monomer B. Once the CPHF coefficients are ready, the SAPT0 induction energy $E^{(200)}_{\rm ind,resp}$ can be computed very easily:
# 
# \begin{equation}
# E^{(200)}_{\rm ind,resp}=4\tilde{v}^{rb}_{ab}C^a_r+4\tilde{v}^{as}_{ab}C^b_s
# \end{equation}
# 
# The call to the `helper_SAPT` function `sapt.chf` generates the corresponding contribution to $E^{(200)}_{\rm ind,resp}$ as a byproduct of the calculation of the CPHF coefficients $C^a_r$/$C^b_s$.
# 


### Start E200 Induction and Exchange-Induction

# E200Induction and CPHF orbitals
ind_timer = sapt_timer('induction')

CPHF_ra, Ind20_ba = sapt.chf('B', ind=True)
sapt_printer('Ind20,r (A<-B)', Ind20_ba)

CPHF_sb, Ind20_ab = sapt.chf('A', ind=True)
sapt_printer('Ind20,r (A->B)', Ind20_ab)

Ind20r = Ind20_ba + Ind20_ab



# # 7. Exchange induction energy
# 
# Just like for induction energy, the SAPT0 exchange-induction energy $E^{(200)}_{\rm exch-ind,resp}$ decomposes into two parts describing the exchange quenching of the polarization of A by B and of the polarization of B by A:
# 
# \begin{equation}
# E^{(200)}_{\rm exch-ind,resp}=E^{(200)}_{\rm exch-ind,resp}({\rm A}\leftarrow{\rm B})+E^{(200)}_{\rm exch-ind,resp}({\rm B}\leftarrow{\rm A})
# \end{equation}
# 
# Now, the formula for the A$\leftarrow$B part is given e.g. by Eq. (5) of [Patkowski:2007]:
# 
# \begin{align}
# E^{(200)}_{\rm exch-ind,resp}({\rm A}\leftarrow {\rm B})=&-2 C^a_r \left[\tilde{v}^{br}_{ab}+2S^b_a\tilde{v}^{a'r}_{a'b}+2S^b_{a'}\tilde{v}^{ra'}_{ab}-S^b_a\tilde{v}^{ra'}_{a'b}-S^b_{a'}\tilde{v}^{a'r}_{ab}+2S^r_{b'}\tilde{v}^{b'b}_{ab}\right.\\ &-S^r_{b'}\tilde{v}^{bb'}_{ab}-2S^b_a S^r_{b'}\tilde{v}^{a'b'}_{a'b}-2S^b_{a'}S^{a'}_{b'}\tilde{v}^{rb'}_{ab}-2S^{b'}_{a'}S^r_{b'}\tilde{v}^{a'b}_{ab}-2S^{b'}_a S^{a'}_{b'}\tilde{v}^{rb}_{a'b}\\ & \left.+S^b_{a'}S^r_{b'}\tilde{v}^{a'b'}_{ab}+S^b_a S^{a'}_{b'}\tilde{v}^{rb'}_{a'b}\right]
# \end{align}
# 
# and the corresponding formula for the B$\leftarrow$A part is obtained by interchanging the symbols pertaining to A with those of B $(a\leftrightarrow b,r\leftrightarrow s)$ in the above expression. In this example, the CPHF coefficients $C^a_r$ and $C^b_s$ obtained in the previous section are combined with *dressed* two-electron integrals and overlap integrals to compute the $E^{(200)}_{\rm exch-ind,resp}$ expression term by term.
# 

# Exchange-Induction

# A <- B
vt_abra = sapt.vt('abra')
vt_abar = sapt.vt('abar')
ExchInd20_ab  =     np.einsum('ra,abbr', CPHF_ra, sapt.vt('abbr'))
ExchInd20_ab += 2 * np.einsum('rA,Ab,abar', CPHF_ra, sapt.s('ab'), vt_abar)
ExchInd20_ab += 2 * np.einsum('ra,Ab,abrA', CPHF_ra, sapt.s('ab'), vt_abra)
ExchInd20_ab -=     np.einsum('rA,Ab,abra', CPHF_ra, sapt.s('ab'), vt_abra)

vt_abbb = sapt.vt('abbb')
vt_abab = sapt.vt('abab')
ExchInd20_ab -=     np.einsum('ra,Ab,abAr', CPHF_ra, sapt.s('ab'), vt_abar)
ExchInd20_ab += 2 * np.einsum('ra,Br,abBb', CPHF_ra, sapt.s('br'), vt_abbb)
ExchInd20_ab -=     np.einsum('ra,Br,abbB', CPHF_ra, sapt.s('br'), vt_abbb)
ExchInd20_ab -= 2 * np.einsum('rA,Ab,Br,abaB', CPHF_ra, sapt.s('ab'), sapt.s('br'), vt_abab)

vt_abrb = sapt.vt('abrb')
ExchInd20_ab -= 2 * np.einsum('ra,Ab,BA,abrB', CPHF_ra, sapt.s('ab'), sapt.s('ba'), vt_abrb)
ExchInd20_ab -= 2 * np.einsum('ra,AB,Br,abAb', CPHF_ra, sapt.s('ab'), sapt.s('br'), vt_abab)
ExchInd20_ab -= 2 * np.einsum('rA,AB,Ba,abrb', CPHF_ra, sapt.s('ab'), sapt.s('ba'), vt_abrb)

ExchInd20_ab +=     np.einsum('ra,Ab,Br,abAB', CPHF_ra, sapt.s('ab'), sapt.s('br'), vt_abab)
ExchInd20_ab +=     np.einsum('rA,Ab,Ba,abrB', CPHF_ra, sapt.s('ab'), sapt.s('ba'), vt_abrb)

ExchInd20_ab *= -2
sapt_printer('Exch-Ind20,r (A<-B)', ExchInd20_ab)

# B <- A
vt_abbs = sapt.vt('abbs')
vt_absb = sapt.vt('absb')
ExchInd20_ba  =     np.einsum('sb,absa', CPHF_sb, sapt.vt('absa'))
ExchInd20_ba += 2 * np.einsum('sB,Ba,absb', CPHF_sb, sapt.s('ba'), vt_absb)
ExchInd20_ba += 2 * np.einsum('sb,Ba,abBs', CPHF_sb, sapt.s('ba'), vt_abbs)
ExchInd20_ba -=     np.einsum('sB,Ba,abbs', CPHF_sb, sapt.s('ba'), vt_abbs)

vt_abaa = sapt.vt('abaa')
vt_abab = sapt.vt('abab')
ExchInd20_ba -=     np.einsum('sb,Ba,absB', CPHF_sb, sapt.s('ba'), vt_absb)
ExchInd20_ba += 2 * np.einsum('sb,As,abaA', CPHF_sb, sapt.s('as'), vt_abaa)
ExchInd20_ba -=     np.einsum('sb,As,abAa', CPHF_sb, sapt.s('as'), vt_abaa)
ExchInd20_ba -= 2 * np.einsum('sB,Ba,As,abAb', CPHF_sb, sapt.s('ba'), sapt.s('as'), vt_abab)

vt_abas = sapt.vt('abas')
ExchInd20_ba -= 2 * np.einsum('sb,Ba,AB,abAs', CPHF_sb, sapt.s('ba'), sapt.s('ab'), vt_abas)
ExchInd20_ba -= 2 * np.einsum('sb,BA,As,abaB', CPHF_sb, sapt.s('ba'), sapt.s('as'), vt_abab)
ExchInd20_ba -= 2 * np.einsum('sB,BA,Ab,abas', CPHF_sb, sapt.s('ba'), sapt.s('ab'), vt_abas)

ExchInd20_ba +=     np.einsum('sb,Ba,As,abAB', CPHF_sb, sapt.s('ba'), sapt.s('as'), vt_abab)
ExchInd20_ba +=     np.einsum('sB,Ba,Ab,abAs', CPHF_sb, sapt.s('ba'), sapt.s('ab'), vt_abas)

ExchInd20_ba *= -2
sapt_printer('Exch-Ind20,r (A->B)', ExchInd20_ba)
ExchInd20r = ExchInd20_ba + ExchInd20_ab

ind_timer.stop()
### End E200 Induction and Exchange-Induction



# # 8. Summary table
# 
# All the SAPT0 interaction energy contributions have been calculated. All that is left to do is to print out the contributions and the total energy, and to compare the results with the SAPT0 corrections calculated directly by Psi4.
# 

print('SAPT0 Results')
print('-' * 70)
sapt_printer('Exch10 (S^2)', Exch100)
sapt_printer('Elst10', Elst10)
sapt_printer('Disp20', Disp200)
sapt_printer('Exch-Disp20', ExchDisp20)
sapt_printer('Ind20,r', Ind20r)
sapt_printer('Exch-Ind20,r', ExchInd20r)

print('-' * 70)
sapt0 = Exch100 + Elst10 + Disp200 + ExchDisp20 + Ind20r + ExchInd20r
sapt_printer('Total SAPT0', sapt0)

# ==> Compare to Psi4 <==
psi4.set_options({'df_basis_sapt':'aug-cc-pvtz-ri'})
psi4.energy('sapt0')
Eelst = psi4.get_variable('SAPT ELST ENERGY')
Eexch = psi4.get_variable('SAPT EXCH10(S^2) ENERGY')
Eind  = psi4.get_variable('SAPT IND20,R ENERGY')
Eexind  = psi4.get_variable('SAPT EXCH-IND20,R ENERGY')
Edisp  = psi4.get_variable('SAPT DISP20 ENERGY')
Eexdisp  = psi4.get_variable('SAPT EXCH-DISP20 ENERGY')
psi4.driver.p4util.compare_values(Eelst, Elst10, 6, 'Elst100')
psi4.driver.p4util.compare_values(Eexch, Exch100, 6, 'Exch100(S^2)')
psi4.driver.p4util.compare_values(Edisp, Disp200, 6, 'Disp200')
psi4.driver.p4util.compare_values(Eexdisp, ExchDisp20, 6, 'Exch-Disp200')
psi4.driver.p4util.compare_values(Eind, Ind20r, 6, 'Ind200,r')
psi4.driver.p4util.compare_values(Eexind, ExchInd20r, 6, 'Exch-Ind200,r')



# ## References
# 
# 1. The classic review paper on SAPT: "Perturbation Theory Approach to Intermolecular Potential Energy Surfaces of van der Waals Complexes"
# 	> [[Jeziorski:1994](http://pubs.acs.org/doi/abs/10.1021/cr00031a008)] B. Jeziorski, R. Moszynski, and K. Szalewicz, *Chem. Rev.* **94**, 1887 (1994)
# 2. The definitions and practical comparison of different levels of SAPT: "Levels of symmetry adapted perturbation theory (SAPT). I. Efficiency and performance for interaction energies"
# 	> [[Parker:2014](http://aip.scitation.org/doi/10.1063/1.4867135)] T. M. Parker, L. A. Burns, R. M. Parrish, A. G. Ryno, and C. D. Sherrill, *J. Chem. Phys.* **140**, 094106 (2014)
# 3. Second-order SAPT exchange corrections without the $S^2$ approximation: "Single-determinant-based symmetry-adapted perturbation theory without single-exchange approximation"
# 	> [[Schaffer:2013](http://www.tandfonline.com/doi/abs/10.1080/00268976.2013.827253)] R. Schäffer and G. Jansen, *Mol. Phys.* **111**, 2570 (2013)
# 4. Alternative, second-quantization based approach to SAPT exchange corrections: "Many‐body theory of exchange effects in intermolecular interactions. Second‐quantization approach and comparison with full configuration interaction results"
# 	> [[Moszynski:1994a](http://aip.scitation.org/doi/abs/10.1063/1.466661)] R. Moszynski, B. Jeziorski, and K. Szalewicz, *J. Chem. Phys.* **100**, 1312 (1994)
# 5. The density-matrix formalism for SAPT exchange corrections employed in this work: "Many‐body theory of exchange effects in intermolecular interactions. Density matrix approach and applications to He–F$^−$, He–HF, H$_2$–HF, and Ar–H$_2$ dimers"
# 	> [[Moszynski:1994b](http://aip.scitation.org/doi/abs/10.1063/1.467225)] R. Moszynski, B. Jeziorski, S. Rybak, K. Szalewicz, and H. L. Williams, *J. Chem. Phys.* **100**, 5080 (1994)
# 6. A classic paper with derivations of many SAPT corrections: "Many‐body symmetry‐adapted perturbation theory of intermolecular interactions. H$_2$O and HF dimers"
# 	> [[Rybak:1991](http://aip.scitation.org/doi/abs/10.1063/1.461528)] S. Rybak, B. Jeziorski, and K. Szalewicz, *J. Chem. Phys.* **95**, 6576 (1991)
# 7. A paper about the frozen-core approximation in SAPT, containing the corrected formula for the exchange dispersion energy: "Frozen core and effective core potentials in symmetry-adapted perturbation theory"
# 	> [[Patkowski:2007](http://aip.scitation.org/doi/10.1063/1.2784391)] K. Patkowski and K. Szalewicz, *J. Chem. Phys.* **127**, 164103 (2007)
# 8. A classic paper about the CPHF equations: "Perturbed Hartree–Fock Theory. I. Diagrammatic Double‐Perturbation Analysis"
# 	> [[Caves:1969](http://aip.scitation.org/doi/abs/10.1063/1.1671609)] T. C. Caves and M. Karplus, *J. Chem. Phys.* **50**, 3649 (1969)
# 




# # MintsHelper: Generating 1- and 2-electron Integrals with <span style='font-variant: small-caps'> Psi4 </span>
# 
# In all of quantum chemistry, one process which is common to nearly every method is the evaluation of one-
# and two-electron integrals.  Fortunately, we can leverage infrastructure in <span style='font-variant: small-caps'> 
# Psi4 </span> to perform this task for us.  This tutorial will discuss the [``psi4.core.MintsHelper``](http://psicode.org/psi4manual/master/psi4api.html#psi4.core.MintsHelper "Go to API") class, which is an
# interface for the powerful Psi4 ``libmints`` library which wraps the `libint` library, where these integrals are actually computed.  
# 
# ## MintsHelper Overview
# In order to compute 1- and 2-electron integrals, we first need a molecule and basis set with which to work.  So, 
# before diving into `MintsHelper`, we need to build these objects.  In the cell below, we have imported
# <span style='font-variant: small-caps'> Psi4 </span> and NumPy, defined a water molecule, and set the basis to
# cc-pVDZ.  We've also set the memory available to <span style='font-variant: small-caps'> Psi4</span>, as well as
# defined a variable `numpy_memory` which we will discuss later.
# 

# ==> Setup <==
# Import statements
import psi4
import numpy as np

# Memory & Output file
psi4.set_memory(int(2e9))
numpy_memory = 2
psi4.core.set_output_file('output.dat', False)

# Molecule definition
h2o = psi4.geometry("""
O
H 1 0.96
H 1 0.96 2 104.5
""")

# Basis Set
psi4.set_options({'basis': 'cc-pvdz'})


# Now, we are ready to create an instance of the `MintsHelper` class.  To do this, we need to pass a `BasisSet`
# object to the `MintsHelper` initializer.  Fortunately, from the previous tutorial on the `Wavefunction` class, we know
# that we can obtain such an object from an existing wavefunction.  So, let's build a new wavefunction for our molecule,
# get the basis set object, and build an instance of `MintsHelper`:
# 

# ==> Build MintsHelper Instance <==
# Build new wavefunction
wfn = psi4.core.Wavefunction.build(h2o, psi4.core.get_global_option('basis'))

# Initialize MintsHelper with wavefunction's basis set
mints = psi4.core.MintsHelper(wfn.basisset())


# Below are summarized several commonly computed quantities and how to obtain them using a `MintsHelper` class method:
# 
# | Quantity | Function | Description |
# |----------|----------|-------------|
# | AO Overlap integrals | [mints.ao_overlap()](http://psicode.org/psi4manual/master/psi4api.html#psi4.core.MintsHelper.ao_overlap "Go to Documentation") | Returns AO overlap matrix as a `psi4.core.Matrix` object |
# | AO Kinetic Energy | [mints.ao_kinetic()](http://psicode.org/psi4manual/master/psi4api.html#psi4.core.MintsHelper.ao_kinetic "Go to Documentation") | Returns AO kinetic energy matrix as a `psi4.core.Matrix` object |
# | AO Potential Energy | [mints.ao_potential()](http://psicode.org/psi4manual/master/psi4api.html#psi4.core.MintsHelper.ao_potential "Go to Documentation") | Returns AO potential energy matrix as a `psi4.core.Matrix` object |
# | AO Electron Repulsion Integrals | [mints.ao_eri()](http://psicode.org/psi4manual/master/psi4api.html#psi4.core.MintsHelper.ao_eri "Go to Documentation") | Returns rank-4 electron repulsion integral tensor in AO basis |
# 
# In addition to these methods, another which is worth mentioning is the [`MintsHelper.mo_eri()`](http://psicode.org
# psi4manual(/master/psi4api.html#psi4.core.MintsHelper.mo_eri, "Go, to, Documentation"), function,, which, can, transform)
# the four-index, two-electron repulsion integrals from the atomic orbital (AO) to the molecular orbital (MO) basis,
# which will be important in MP2 theory.  
# 
# ## Memory Considerations
# 
# Before moving forward to computing any 1- or 2-electron integrals, we must first discuss the memory requirements of
# these objects.  Whenever these quantities are computed, they are stored directly in memory (a.k.a. RAM,
# *not* on the hard drive) which, for a typical laptop or personal computer, usually tops out at around 16 GB of 
# space.  The storage space required by the two-index AO overlap integrals and four-index ERIs scales as ${\cal
# O}(N^2)$ and ${\cal O}(N^4)$, respectively, where $N$ is the number of AO basis functions.  This means that for a
# system with 500 AO basis functions, while the AO overlap integrals will only require 1 MB of memory to store,
# the ERIs will require a staggering **500 GB** of memory!! This can be reduced to **62.5 GB** of memory if integral permutational symmetry is used. For this reason, as well as the steep computational 
# scaling of many of the methods demonstrated here, we limit ourselves to small systems ($\sim50$ basis functions)
# which should not require such egregious amounts of memory.  Additionally, we will employ a "memory check" to catch
# any case which could potentially try to use more memory than is available:
# ~~~python
# # Memory check for ERI tensor
# I_size = (nbf**4) * 8.e-9
# print('Size of the ERI tensor will be %4.2f GB.' % (I_size))
# memory_footprint = I_size * 1.5
# if I_size > numpy_memory:
#     psi4.core.clean()
#     raise Exception("Estimated memory utilization (%4.2f GB) exceeds allotted memory                      limit of %4.2f GB." % (memory_footprint, numpy_memory))
# ~~~
# Using the `numpy_memory` variable, we are able to control whether the ERIs will be computed, based on the amount of
# memory required to store them. 
# 
# <font color="red">**NOTE: DO NOT EXCEED YOUR SYSTEM'S MEMORY.  THIS MAY RESULT IN YOUR PROGRAM AND/OR COMPUTER CRASHING!**</font>
# 
# ## Examples: AO Overlap, AO ERIs, Core Hamiltonian
# The cell below demonstrates obtaining the AO overlap integrals, conducting the
# above memory check, and computing the ERIs and core Hamiltonian matrix for our water molecule.
# 

# ==> Integrals galore! <==
# AO Overlap
S = np.asarray(mints.ao_overlap())

# Number of basis functions
nbf = S.shape[0]

# Memory check
I_size = (nbf ** 4) * 8.e-9
print('Size of the ERI tensor will be %4.2f GB.' % (I_size))
memory_footprint = I_size * 1.5
if I_size > numpy_memory:
    psi4.core.clean()
    raise Exception("Estimated memory utilization (%4.2f GB) exceeds allotted memory                      limit of %4.2f GB." % (memory_footprint, numpy_memory))

# Compute AO-basis ERIs
I = mints.ao_eri()

# Compute AO Core Hamiltonian
T = np.asarray(mints.ao_kinetic())
V = np.asarray(mints.ao_potential())
H = T + V


# # Density Functional Theory: Grid
# ## I. Theoretical Overview
# This tutorial will discuss the basics of DFT and discuss the grid used to evaluate DFT quantities.
# As with HF, DFT aims to solve the generalized eigenvalue problem:
# 
# $$\sum_{\nu} F_{\mu\nu}C_{\nu i} = \epsilon_i\sum_{\nu}S_{\mu\nu}C_{\nu i}$$
# $${\bf FC} = {\bf SC\epsilon},$$
# 
# Where with HF the Fock matrix is constructed as:
# 
# $$F^{HF}_{\mu\nu} = H_{\mu\nu} + 2J[D]_{\mu\nu} - K[D]_{\mu\nu}$$
# 
# $$D_{\mu\nu} = C_{\mu i} C_{\nu i}$$
# 
# With DFT we generalize this construction slightly to:
# $$F^{DFT}_{\mu\nu} = H_{\mu\nu} + 2J[D]_{\mu\nu} - \zeta K[D]_{\mu\nu} + V^{\rm{xc}}_{\mu\nu}$$
# 
# $\zeta$ is an adjustable parameter where we can very the amount of exact (HF) exchange and $V$ is the DFT potenital which typically attempts to add dynamical correlation in the self-consistent field methodolgy.
# 
# 

# ## 2. Examining the Grid
# We will discuss the evaluation and manipulation of the grid.
# 

import psi4
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
get_ipython().magic('matplotlib notebook')


# Set computatation options and molecule, any single atom will do.

mol = psi4.geometry("He")
psi4.set_options({'BASIS':                'CC-PVDZ',
                  'DFT_SPHERICAL_POINTS': 50,
                  'DFT_RADIAL_POINTS':    12})


basis = psi4.core.BasisSet.build(mol, "ORBITAL", "CC-PVDZ")
sup = psi4.driver.dft_funcs.build_superfunctional("PBE", True)[0]
Vpot = psi4.core.VBase.build(basis, sup, "RV")
Vpot.initialize()


x, y, z, w = Vpot.get_np_xyzw()
R = np.sqrt(x **2 + y ** 2 + z **2)


fig, ax = plt.subplots()
ax.scatter(x, y, c=w)
#ax.set_xscale('log')


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
mask = R > 8
p = ax.scatter(x[mask], y[mask], z[mask], c=w[mask], marker='o')
plt.colorbar(p)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')


mol = psi4.geometry("""
 O
 H 1 1.1
 H 1 1.1 2 104
""")
mol.update_geometry()
psi4.set_options({'BASIS': '              CC-PVDZ',
                  'DFT_SPHERICAL_POINTS': 26,
                  'DFT_RADIAL_POINTS':    12})

basis = psi4.core.BasisSet.build(mol, "ORBITAL", "CC-PVDZ")
sup = psi4.driver.dft_funcs.build_superfunctional("PBE", True)[0]
Vpot = psi4.core.VBase.build(basis, sup, "RV")
Vpot.initialize()
x, y, z, w = Vpot.get_np_xyzw()
R = np.sqrt(x **2 + y ** 2 + z **2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
mask = R > 0
p = ax.scatter(x[mask], y[mask], z[mask], c=w[mask], marker='o')
plt.colorbar(p)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')


# ## Refs:
# - Koch, W. and Holthausen, M.C., **2001**, A Chemist’s Guide to Density Functional Theory, 2nd, Wiley-VCH, Weinheim.
# - Kohn, W. and Sham, L. *J, Phys. Rev.*, **1965**, *140*, A1133- A1138
# - Becke, A.D., *J. Chem. Phys.*, **1988**, *88*, 2547
# - Treutler, O. and Ahlrichs, R., *J. Chem. Phys.*, **1995**, *102*, 346
# - Gill, P.M.W., Johnson, B.G., and Pople, J.A., *Chem. Phys. Lett.*, **1993,209 (5), pp. 506, 16 July 1993.
# 




# # Tensor Manipulation: Psi4 and NumPy manipulation routines
# Contracting tensors together forms the core of the Psi4NumPy project. First let us consider the popluar [Einstein Summation Notation](https://en.wikipedia.org/wiki/Einstein_notation) which allows for very succinct descriptions of a given tensor contraction.
# 
# For example, let us consider a [inner (dot) product](https://en.wikipedia.org/wiki/Dot_product):
# $$c = \sum_{ij} A_{ij} * B_{ij}$$
# 
# With the Einstein convention, all indices that are repeated are considered summed over, and the explicit summation symbol is dropped:
# $$c = A_{ij} * B_{ij}$$
# 
# This can be extended to [matrix multiplication](https://en.wikipedia.org/wiki/Matrix_multiplication):
# \begin{align}
# \rm{Conventional}\;\;\;  C_{ik} &= \sum_{j} A_{ij} * B_{jk} \\rm{Einstein}\;\;\;  C &= A_{ij} * B_{jk} \\end{align}
# 
# Where the $C$ matrix has *implied* indices of $C_{ik}$ as the only repeated index is $j$.
# 
# However, there are many cases where this notation fails. Thus we often use the generalized Einstein convention. To demonstrate let us examine a [Hadamard product](https://en.wikipedia.org/wiki/Hadamard_product_(matrices)):
# $$C_{ij} = \sum_{ij} A_{ij} * B_{ij}$$
# 
# 
# This operation is nearly identical to the dot product above, and is not able to be written in pure Einstein convention. The generalized convention allows for the use of indices on the left hand side of the equation:
# $$C_{ij} = A_{ij} * B_{ij}$$
# 
# Usually it should be apparent within the context the exact meaning of a given expression.
# 
# Finally we also make use of Matrix notation:
# \begin{align}
# {\rm Matrix}\;\;\;  \bf{D} &= \bf{A B C} \{\rm Einstein}\;\;\;  D_{il} &= A_{ij} B_{jk} C_{kl}
# \end{align}
# 
# Note that this notation is signified by the use of bold characters to denote matrices and consecutive matrices next to each other imply a chain of matrix multiplications! 
# 

# ## Einsum
# 
# To perform most operations we turn to [NumPy's einsum function](https://docs.scipy.org/doc/numpy/reference/generated/numpy.einsum.html) which allows the Einsten convention as an input. In addition to being much easier to read, manipulate, and change, it is also much more efficient that a pure Python implementation.
# 
# To begin let us consider the construction of the following tensor (which you may recognize):
# $$G_{pq} = 2.0 * I_{pqrs} D_{rs} - 1.0 * I_{prqs} D_{rs}$$ 
# 
# First let us import our normal suite of modules:
# 

import numpy as np
import psi4
import time


# We can then use conventional Python loops and einsum to perform the same task. Keep size relatively small as these 4-index tensors grow very quickly in size.
# 

size = 20

if size > 30:
    raise Exception("Size must be smaller than 30.")
D = np.random.rand(size, size)
I = np.random.rand(size, size, size, size)

# Build the fock matrix using loops, while keeping track of time
tstart_loop = time.time()
Gloop = np.zeros((size, size))
for p in range(size):
    for q in range(size):
        for r in range(size):
            for s in range(size):
                Gloop[p, q] += 2 * I[p, q, r, s] * D[r, s]
                Gloop[p, q] -=     I[p, r, q, s] * D[r, s]

g_loop_time = time.time() - tstart_loop

# Build the fock matrix using einsum, while keeping track of time
tstart_einsum = time.time()
J = np.einsum('pqrs,rs', I, D)
K = np.einsum('prqs,rs', I, D)
G = 2 * J - K

einsum_time = time.time() - tstart_einsum

# Make sure the correct answer is obtained
print('The loop and einsum fock builds match:    %s\n' % np.allclose(G, Gloop))
# Print out relative times for explicit loop vs einsum Fock builds
print('Time for loop G build:   %14.4f seconds' % g_loop_time)
print('Time for einsum G build: %14.4f seconds' % einsum_time)
print('G builds with einsum are {:3.4f} times faster than Python loops!'.format(g_loop_time / einsum_time))


# As you can see, the einsum function is considerably faster than the pure Python loops and, in this author's opinion, much cleaner and easier to use.
# 

# ## Dot
# 
# Now let us turn our attention to a more canonical matrix multiplication example such as:
# $$D_{il} = A_{ij} B_{jk} C_{kl}$$
# 
# We could perform this operation using einsum; however, matrix multiplication is an extremely common operation in all branches of linear algebra. Thus, these functions have been optimized to be more efficient than the `einsum` function. The matrix product will explicitly compute the following operation:
# $$C_{ij} = A_{ij} * B_{ij}$$
# 
# This can be called with [NumPy's dot function](https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html#numpy.dot).
# 

size = 200
A = np.random.rand(size, size)
B = np.random.rand(size, size)
C = np.random.rand(size, size)

# First compute the pair product
tmp_dot = np.dot(A, B)
tmp_einsum = np.einsum('ij,jk->ik', A, B)
print("Pair product allclose: %s" % np.allclose(tmp_dot, tmp_einsum))


# Now that we have proved exactly what the dot product does, let us consider the full chain and do a timing comparison:
# 

D_dot = np.dot(A, B).dot(C)
D_einsum = np.einsum('ij,jk,kl->il', A, B, C)
print("Chain multiplication allclose: %s" % np.allclose(D_dot, D_einsum))

print("\nnp.dot time:")
get_ipython().magic('timeit np.dot(A, B).dot(C)')

print("\nnp.einsum time")
get_ipython().magic("timeit np.einsum('ij,jk,kl->il', A, B, C)")


# On most machines the `np.dot` times are roughly ~3,000 times faster. The reason is twofold:
#  - The `np.dot` routines typically call [Basic Linear Algebra Subprograms (BLAS)](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms). The BLAS routines are highly optimized and threaded versions of the code.
#  - The `np.einsum` code will not factorize the operation; Thus, the overall cost is ${\cal O}(N^4)$ (as there are four indices) rather than the factored $(\bf{A B}) \bf{C}$ which runs ${\cal O}(N^3)$.
#  
# The first issue is difficult to overcome; however, the second issue can be resolved by the following:
# 

print("np.einsum factorized time:")
get_ipython().magic("timeit np.einsum('ik,kl->il', np.einsum('ij,jk->ik', A, B), C)")


# On most machines the factorized `einsum` expression is only ~20 times slower than `np.dot`. While a massive improvement, this is a clear demonstration the BLAS usage is usually recommended. It is a tradeoff between speed and readability. The Psi4NumPy project tends to lean toward `einsum` usage except in case where the benefit is too large to pass up.
# 
# It should be noted that in NumPy 1.12 the [einsum function](https://docs.scipy.org/doc/numpy/reference/generated/numpy.einsum.html) has a `optimize` flag which will automatically factorize the einsum code for you. However, NumPy 1.12 was recently released and is not in most installations yet.
# 

# ## Complex tensor manipulations
# Let us consider a popular index transformation example:
# $$M_{pqrs} = C_{pi} C_{qj} I_{ijkl} C_{rk} C_{sl}$$
# 
# Here, a naive `einsum` call would scale like $\mathcal{O}(N^8)$ which translates to an extremely costly computation for all but the smallest $N$.
# 

# Grab orbitals
size = 15
if size > 15:
    raise Exception("Size must be smaller than 15.")
    
C = np.random.rand(size, size)
I = np.random.rand(size, size, size, size)

# Numpy einsum N^8 transformation.
print("\nStarting Numpy's N^8 transformation...")
n8_tstart = time.time()
MO_n8 = np.einsum('pI,qJ,pqrs,rK,sL->IJKL', C, C, I, C, C)
n8_time = time.time() - n8_tstart
print("...transformation complete in %.3f seconds." % (n8_time))

# Numpy einsum N^5 transformation.
print("\n\nStarting Numpy's N^5 transformation with einsum...")
n5_tstart = time.time()
MO_n5 = np.einsum('pA,pqrs->Aqrs', C, I)
MO_n5 = np.einsum('qB,Aqrs->ABrs', C, MO_n5)
MO_n5 = np.einsum('rC,ABrs->ABCs', C, MO_n5)
MO_n5 = np.einsum('sD,ABCs->ABCD', C, MO_n5)
n5_time = time.time() - n5_tstart
print("...transformation complete in %.3f seconds." % n5_time)
print("\nN^5 %4.2f faster than N^8 algorithm!" % (n8_time / n5_time))
print("Allclose: %s" % np.allclose(MO_n8, MO_n5))

# Numpy GEMM N^5 transformation.
# Try to figure this one out!
print("\n\nStarting Numpy's N^5 transformation with dot...")
dgemm_tstart = time.time()
MO = np.dot(C.T, I.reshape(size, -1))
MO = np.dot(MO.reshape(-1, size), C)
MO = MO.reshape(size, size, size, size).transpose(1, 0, 3, 2)

MO = np.dot(C.T, MO.reshape(size, -1))
MO = np.dot(MO.reshape(-1, size), C)
MO = MO.reshape(size, size, size, size).transpose(1, 0, 3, 2)
dgemm_time = time.time() - dgemm_tstart
print("...transformation complete in %.3f seconds." % dgemm_time)
print("\nAllclose: %s" % np.allclose(MO_n8, MO))
print("N^5 %4.2f faster than N^8 algorithm!" % (n8_time / dgemm_time))





# # Psi4 $\leftrightarrow$ NumPy Data Sharing
# 
# The heart of the Psi4NumPy project its the ability to easily share and manipulate quantities in Python. While Psi4 offers the ability to manipulate most objects and perform tensor operations at the Python layer, it is often much easier to use the NumPy project, as its focus is on ease of use rather than optimal performance. Fortunately, Psi4 offers seemless integration with the NumPy framework. More details on the underlying functions can be found in the Psi4 [documentation](http://psicode.org/psi4manual/master/numpy.html).
# 
# As before, let us start off with importing Psi4 and NumPy while also creating a random `5 x 5` NumPy array:
# 

import psi4
import numpy as np

# Random number array
array = np.random.rand(5, 5)


# Converting this to a Psi4 Matrix, which is an instance of the [`psi4.core.Matrix`](http://psicode.org/psi4manual/master/psi4api.html#psi4.core.Matrix 
# "Go to API") class, and back again is as simple as:
# 

psi4_matrix = psi4.core.Matrix.from_array(array)
new_array = np.array(psi4_matrix)

print("Allclose new_array, array:", np.allclose(new_array, array))


# ## Views
# Because both of these objects have the same in-memory data layout, the conversion is accomplished through the NumPy 
# [array_interface](https://docs.scipy.org/doc/numpy/reference/arrays.interface.html). This also opens the opportunity 
# to manipulate the Psi4 Matrix and Vector classes directly in memory.  To do this, we employ the `.np` attribute:
# 

matrix = psi4.core.Matrix(3, 3)
print("Zero Psi4 Matrix:")
print(np.array(matrix))

matrix.np[:] = 1
print("\nMatrix updated to ones:")
print(np.array(matrix))


# The `.np` attribute effectively returns a NumPy [view](http://scipy-cookbook.readthedocs.io/items/ViewsVsCopies.html). This view can then be manipulated as a conventional NumPy array and the underlying Psi4 Matrix data will be modified.
# 
# <font color='red'>**Warning!** The following operation operation is incorrect and can potenitally lead to confusion:</font>
# 

print(psi4.core.Matrix(3, 3).np)


# While the above operation works about ~90% of the time, occasionally you will notice extremely large and small values. This is due to the fact that when you create the Psi4 Matrix and grab its view, the Psi4 Matrix is no longer bound to anything, and Python will attempt to "garbage collect" or remove the object. This sometimes happens *before* Python prints out the object so the NumPy view is pointing to a random piece of data in memory. A safe way to do this would be:
# 

mat = psi4.core.Matrix(3, 3)
print(mat.np)

# or
print(np.asarray(psi4.core.Matrix(3, 3)))


# Similar to the `.np` attribute, one can use `np.asarray` to create a NumPy view of a Psi4 object. Keep in mind that this is different than `np.array` which will copy the data.
# 

mat = psi4.core.Matrix(3, 3)
mat_view = np.asarray(mat)

mat_view[:] = np.random.random(mat.shape)
print(mat.np)


# Keep in mind that you must *update* this view using the `[]` syntax and not replace it (`=`). The following example should demonstrate the difference:
# 

mat_view = np.zeros((3, 3))

# Mat is not updated as we replaced the mat_view with a new NumPy matrix.
print(mat.np)


# ## Vector class
# Like the Psi4 Matrix class, the [`psi4.core.Vector`](http://psicode.org/psi4manual/master/psi4api.html#psi4.core.Vector "Go to API")
# class has similar accessors:
# 

arr = np.random.rand(5)
vec = psi4.core.Vector.from_array(arr)
print(vec.np)


"""Tutorial: Second-Order Moller--Plesset Perturbation Theory (MP2)"""

__author__    = "Dominic A. Sirianni"
__credit__    = ["Dominic A. Sirianni", "Daniel G. A. Smith"]

__copyright__ = "(c) 2014-2017, The Psi4NumPy Developers"
__license__   = "BSD-3-Clause"
__date__      = "2017-05-23"


# # Second-Order Moller-Plesset Perturbation Theory (MP2)
# 
# Moller-Plesset perturbation theory [also referred to as many-body perturbation theory (MBPT)] is an adaptation of the more general Rayleigh-Schrodinger perturbation theory (RSPT), applied to problems in molecular electronic structure theory.  This tutorial will provide a brief overview of both RSPT and MBPT, before walking through an implementation of second-order Moller-Plesset perturbation theory (specifically referred to as MP2) which uses conventional, 4-index ERIs.  
# 
# ### I. Overview of Rayleigh-Schrodinger Perturbation Theory
# Given the Hamiltonian operator $\hat{H}$ for a system, perturbation theory solves the Schrodinger equation for that system by rewriting $\hat{H}$ as
# 
# \begin{equation}
# \hat{H} = \hat{H}{}^{(0)} + \lambda\hat{V},
# \tag{[Szabo:1996], pp. 322, Eqn. 6.3}
# \end{equation}
# 
# were $\hat{H}{}^{(0)}$ is the Hamiltonian operator corresponding to a solved problem which resembles $\hat{H}$, and $\hat{V}$ is the *perturbation* operator, defined as $\hat{V} = \hat{H} - \hat{H}{}^{(0)}$.  Then the Schrodinger equation for the system becomes 
# 
# \begin{equation}
# \hat{H}\mid\Psi_n\rangle = (\hat{H}{}^{(0)} + \lambda\hat{V})\mid\Psi_n\rangle = E_n\mid\Psi_n\rangle.
# \tag{[Szabo:1996], pp. 322, Eqn. 6.2}
# \end{equation}
# 
# The energies $E_n$ and wavefunctions $\mid\Psi_n\rangle$ will both be functions of $\lambda$; they can therefore be written as a Taylor series expansion about $\lambda = 0$ ([Szabo:1996], pp. 322, Eqns. 6.4a & 6.4b):
# 
# \begin{align}
# E_n &= E_n^{(0)} + \lambda E_n^{(1)} + \lambda^2E_n^{(2)} + \ldots;\tag{[Szabo:1996], pp. 322, Eqn. 6.4a}\\mid\Psi_n\rangle &= \mid\Psi_n^{(0)}\rangle + \lambda\mid\Psi_n^{(1)}\rangle + \lambda^2\mid\Psi_n^{(2)}\rangle + \ldots,\tag{[Szabo:1996], pp. 322, Eqn. 6.4b}\\end{align}
# 
# in practice, these perturbation expansions may be truncated to a given power of $\lambda$.  Substituting the perturbation series above back into the Schrodinger equation yields
# 
# \begin{equation*}
# (\hat{H}{}^{(0)} + \lambda\hat{V})(\mid\Psi_n^{(0)}\rangle + \lambda\mid\Psi_n^{(1)}\rangle + \lambda^2\mid\Psi_n^{(2)}\rangle + \ldots) = (E_n^{(0)} + \lambda E_n^{(1)} + \lambda^2E_n^{(2)} + \ldots)(\mid\Psi_n^{(0)}\rangle + \lambda\mid\Psi_n^{(1)}\rangle + \lambda^2\mid\Psi_n^{(2)}\rangle + \ldots),
# \end{equation*}
# 
# which by equating powers of $\lambda$ ([Szabo:1996], pp. 323, Eqns. 6.7a-6.7d) gives expressions for the $E_n^{(i)}$and $\mid\Psi_n^{(i)}\rangle$. Note that for $\lambda^0$, $E_n^{(0)}$ and $\mid\Psi_n^{(0)}\rangle$ are known, as they are the solution to the zeroth-order problem $\hat{H}{}^{(0)}$.  For $\lambda^1$ and $\lambda^2$, the expressions for $E_n^{(1)}$ and $E_n^{(2)}$ are given by
# 
# \begin{align}
# \lambda^1:\;\;\;\;E_n^{(1)} &= \langle\Psi_n^{(0)}\mid\hat{V}\mid\Psi_n^{(0)}\rangle,\tag{[Szabo:1996], pp. 323, Eqn. 6.8b}\\lambda^2:\;\;\;\;E_n^{(2)} &= \sum_{\mu\neq n}\frac{\mid\langle\Psi_{n}^{(0)}\mid\hat{V}\mid\Psi_{\mu}^{(0)}\rangle\mid^2}{E_n^{(0)} - E_{\mu}^{(0)}}\tag{[Szabo:1996], pp. 324, Eqn. 6.12}\\end{align}
# 

# ### II. Overview of Moller-Plesset Perturbation Theory
# 
# The exact electronic Hamiltonian for an N-electron molecule with a given atomic configuration is (in atomic units):
# 
# \begin{equation}
# \hat{H}_{elec} = \sum_i\hat{h}(i) + \sum_{i<j}\frac{1}{r_{ij}}.\tag{[Szabo:1996], pp. 43, Eqn. 2.10}
# \end{equation}
# 
# Moller-Plesset perturbation theory seeks to solve the molecular electronic Schrodinger equation with the above Hamiltonian using the techniques of Rayleigh-Schroding perturbation theory, by selecting the zeroth-order Hamiltonian and wavefunctions to be those from Hartree-Fock theory:
# 
# \begin{align}
# \hat{H}{}^{(0)}  &= \sum_i \hat{f}(i) = \sum_i \hat{h}(i) + \upsilon^{HF}(i)\tag{[Szabo:1996], pp. 350, Eqn. 6.59}\\hat{V} &= \hat{H} - \hat{H}{}^{(0)} = \sum_{i<j} \frac{1}{r_{ij}} - \upsilon^{HF}(i).\tag{[Szabo:1996], pp. 350, Eqn. 6.60}\\end{align}
# 
# With these choices of $\hat{H}{}^{(0)}$, $\hat{V}$, and $\mid\Psi_n^{(0)}\rangle$, the ground-state zeroth-order energy is given by $E_0^{(0)} = \sum_i\epsilon_i$ ([Szabo:1996], pp. 351, Eqn. 6.67).  Then, the first-order ground state energy may be computed by $E_0^{(1)} = \langle\Psi_0^{HF}\mid\hat{V}\mid\Psi_0^{HF}\rangle$ to find that the total ground-state energy through first order, $E_0 = E_0^{(0)} + E_0^{(1)}$, is exactly the Hartree-Fock energy. ([Szabo:1996], pp. 351, Eqn. 6.69)  Therefore, the first correction to the Hartree-Fock energy will occur at second-order in the perturbation series, i.e., with $E_0^{(2)}$; truncating the perturbation series at second order is commonly referred to as MP2.  The second order correction to the ground state energy will be given by
# 
# \begin{equation}
# E_0^{(2)} = \sum_{\mu\neq 0}\frac{\left|\langle\Psi_{0}^{HF}\mid\hat{V}\mid\Psi_{\mu}^{HF}\rangle\right|^2}{E_0^{(0)} - E_{\mu}^{(0)}}\tag{[Szabo:1996], pp. 351, Eqn. 6.70}
# \end{equation}
# 
# For brevity, we will now drop the "HF" from all zeroth-order wavefunctions.  This summation is over the eigenstate index $\mu$, each associated with a different eigenstate of the zeroth-order Hamiltonian.  For a single-determinant wavefunction constructed from spin orbitals, the summation over the eigenstate index $\mu\neq 0$ therefore refers to determinants which are constructed from *different* spin orbitals than the ground state determinant.  To distinguish such determinants, we will denote MOs occupied in the ground state with indices $i,\,j,\,\ldots$, and MOs which are unoccupied in the ground state with indices $a,\,b,\,\ldots\,$.  Then a determinant where orbital $a$ is substituted for orbital $i$ is denoted $\mid\Psi_i^a\rangle$, and so on.  Before substituting this new notation into the above energy expression, however, we may immediately recognize that many terms $\langle\Psi_{0}\mid\hat{V}\mid\Psi_{\mu}\rangle$ will not contribute to the second order energy:
# 
# | Term         | Determinant                               |  Contribution to $E_0^{(2)}$                            |
# |--------------|-------------------------------------------|---------------------------------------------------------|
# | Singles      | $\mid\Psi_i^a\rangle$                     | 0; Brillouin's Theorem                                  |
# | Doubles      | $\mid\Psi_{ij}^{ab}\rangle$               | Survive                                                 |
# | Higher-order | $\mid\Psi_{ijk\ldots}^{abc\ldots}\rangle$ | 0; $\hat{V}$ is a two-particle operator            |
# 
# Hence we see that only doubly-substituted determinants $\mid\Psi_{ij}^{ab}\rangle$ will contribute to $E_0^{(2)}$.  From Hartree-Fock theory, we know that 
# 
# \begin{equation}
# \langle\Psi_0\mid r_{ij}^{-1}\mid\Psi_{ij}^{ab}\rangle = [ia\| jb],\tag{[Szabo:1996], pp. 72, Tbl. 2.6}
# \end{equation}
# 
# where $[ia\| jb] = [ia\mid jb] - [ib\mid ja]$ is an antisymmetrized two-electron integral, and the square brackets "$[\ldots ]$" indicate that we are employing chemists' notation.  What about the energies of these doubly substituted determinants?  Recognizing that the difference between the energies of the newly- and formerly-occupied orbitals in each substitution must modify the total energy of the ground state determinant, 
# 
# \begin{equation}
# E_{ij}^{ab} = E_0 - (\epsilon_i - \epsilon_a + \epsilon_j - \epsilon_b).
# \end{equation}
# 
# Substituting these expressions into the one for the second-order energy, we have that
# 
# \begin{equation}
# E_0^{(2)} = \sum_{i<j}\sum_{a<b} \frac{\left|\,[ia\|jb]\,\right|^2}{\epsilon_i - \epsilon_a + \epsilon_j - \epsilon_b}.\tag{[Szabo:1996], pp. 352, Eqn. 6.71}
# \end{equation}
# 
# So far, our discussion has used spin-orbitals instead of the more familiar spatial orbitals.  Indeed, significant speedups are achieved when using spatial orbitals.  Integrating out the spin variable $\omega$ from $E_0^{(2)}$ yields two expressions; one each for the interaction of particles with the same spin (SS) and opposite spin (OS):
# 
# \begin{align}
# E_{\rm 0,\,SS}^{(2)} = \sum_{ij}\sum_{ab}\frac{(ia\mid jb)[(ia\mid jb) - (ib\mid ja)]}{\epsilon_i - \epsilon_a + \epsilon_j - \epsilon_b},
# E_{\rm 0,\,OS}^{(2)} = \sum_{ij}\sum_{ab}\frac{(ia\mid jb)(ia\mid jb)}{\epsilon_i - \epsilon_a + \epsilon_j - \epsilon_b},
# \end{align}
# 
# where these spin-free expressions make use of integrals in chemists' notation over spatial orbitals. (Rearranged from [Szabo:1996], pp. 352, Eqn. 6.74)  Note that an exchange integral arises between particles of the same spin; this is because the motions of particles with identical spins are correlated due to the requirement that $\left|\Psi\right|^2$ remain invariant to the exchange of the spatial and spin coordinates of any pair of electrons.  Finally, the total MP2 correction energy $E_0^{(2)} = E_{\rm 0,\,SS}^{(2)} + E_{\rm 0,\,OS}^{(2)}$.
# 
# ### Implementation of Conventional MP2
# 
# Let's begin by importing Psi4 and NumPy, and setting memory and output file options:
# 

# ==> Import statements & Global Options <==
import psi4
import numpy as np

psi4.set_memory(int(2e9))
numpy_memory = 2
psi4.core.set_output_file('output.dat', False)


# Next, we can define our molecule and Psi4 options.  Notice that we are using `scf_type pk` to indicate that we wish to use conventional, full 4-index ERIs, and that we have specified `mp2_type conv` so that the MP2 algorithm we check against also uses the conventional ERIs.
# 

# ==> Molecule & Psi4 Options Definitions <==
mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")


psi4.set_options({'basis':        '6-31g',
                  'scf_type':     'pk',
                  'mp2_type':     'conv',
                  'e_convergence': 1e-8,
                  'd_convergence': 1e-8})


# Since MP2 is a perturbation on the zeroth-order Hartree-Fock description of a molecular system, all of the relevant information (Fock matrix, orbitals, orbital energies) about the system can be computed using any Hartree-Fock program.  We could use the RHF program that we wrote in tutorial 3a, but we could just as easily use Psi4 to do our dirty work.  In the cell below, use Psi4 to compute the RHF energy and wavefunction, and store them using the `return_wfn=True` keyword argument to `psi4.energy()`:
# 

# Get the SCF wavefunction & energies
scf_e, scf_wfn = psi4.energy('scf', return_wfn=True)


# In the expression for $E_0^{(2)}$, the two summations are over occupied and virtual indices, respectively.  Therefore, we'll need to get the number of occupied orbitals and the total number of orbitals.  Additionally, we must obtain the MO energy eigenvalues; again since the sums are over occupied and virtual orbitals, it is good to separate the occupied orbital energies from the virtual orbital energies.  From the SCF wavefunction you generated above, get the number of doubly occupied orbitals, number of molecular orbitals, and MO energies:
# 

# ==> Get orbital information & energy eigenvalues <==
# Number of Occupied orbitals & MOs
ndocc = scf_wfn.nalpha()
nmo = scf_wfn.nmo()

# Get orbital energies, cast into NumPy array, and separate occupied & virtual
eps = np.asarray(scf_wfn.epsilon_a())
e_ij = eps[:ndocc]
e_ab = eps[ndocc:]


# Unlike the orbital information, Psi4 does not return the ERIs when it does a computation.  Fortunately, however, we can just build them again using the `psi4.core.MintsHelper()` class.  Recall that these integrals will be generated in the AO basis; before using them in the $E_0^{(2)}$ expression, we must transform them into the MO basis.  To do this, we first need to obtain the orbital coefficient matrix, **C**.  In the cell below, generate the ERIs for our molecule, get **C** from the SCF wavefunction, and obtain occupied- and virtual-orbital slices of **C** for future use. 
# 

# ==> ERIs <==
# Create instance of MintsHelper class
mints = psi4.core.MintsHelper(scf_wfn.basisset())

# Memory check for ERI tensor
I_size = (nmo**4) * 8.e-9
print('\nSize of the ERI tensor will be %4.2f GB.' % I_size)
memory_footprint = I_size * 1.5
if I_size > numpy_memory:
    psi4.core.clean()
    raise Exception("Estimated memory utilization (%4.2f GB) exceeds allotted memory                      limit of %4.2f GB." % (memory_footprint, numpy_memory))

# Build ERI Tensor
I = np.asarray(mints.ao_eri())

# Get MO coefficients from SCF wavefunction
C = np.asarray(scf_wfn.Ca())
Cocc = C[:, :ndocc]
Cvirt = C[:, ndocc:]


# In order to transform the four-index integrals from the AO to the MO basis, we must perform the following contraction:
# 
# $$(i\,a\mid j\,b) = C_{\mu i}C_{\nu a}(\mu\,\nu\mid|\,\lambda\,\sigma)C_{\lambda j}C_{\sigma b}$$
# 
# Again, here we are using $i,\,j$ as occupied orbital indices and $a,\, b$ as virtual orbital indices.  We could carry out the above contraction all in one step using either `np.einsum()` or explicit loops:
# 
# ~~~python
# # Naive Algorithm for ERI Transformation
# Imo = np.einsum('pi,qa,pqrs,rj,sb->iajb', Cocc, Cvirt, I, Cocc, Cvirt)
# ~~~
# 
# Notice that the transformation from AO index to occupied (virtual) MO index requires only the occupied (virtual) block of the **C** matrix; this allows for computational savings in large basis sets, where the virtual space can be very large.  This algorithm, while simple, has the distinct disadvantage of horrendous scaling.  Examining the contraction more closely, we see that there are 8 unique indices, and thus the step above scales as ${\cal O}(N^8)$.  With this algorithm, a twofold increase of the number of MO's would result in $2^8 = 256\times$ expense to perform.  We can, however, refactor the above contraction such that
# 
# $$(i\,a\mid j\,b) = \left[C_{\mu i}\left[C_{\nu a}\left[C_{\lambda j}\left[C_{\sigma b}(\mu\,\nu\mid|\,\lambda\,\sigma)\right]\right]\right]\right],$$
# 
# where we have now written the transfomation as four ${\cal O}(N^5)$ steps instead of one ${\cal O}(N^8)$ step. This is a savings of $\frac{4}{n^3}$, and is responsible for the feasibility of the MP2 method for application to any but very small systems and/or basis sets.  We may carry out the above ${\cal O}(N^5)$ algorithm by carrying out one index transformation at a time, and storing the result in a temporary array.  In the cell below, transform the ERIs from the AO to MO basis, using our smarter algorithm:
# 

# ==> Transform I -> I_mo @ O(N^5) <==
tmp = np.einsum('pi,pqrs->iqrs', Cocc, I)
tmp = np.einsum('qa,iqrs->iars', Cvirt, tmp)
tmp = np.einsum('iars,rj->iajs', tmp, Cocc)
I_mo = np.einsum('iajs,sb->iajb', tmp, Cvirt)


# We note here that we can use infrastructure in Psi4 to carry out the above integral transformation; this entails obtaining the occupied and virtual blocks of **C** Psi4-side, and then using the built-in `MintsHelper` function `MintsHelper.mo_eri()` to transform the integrals.  Just to check your work above, execute the next cell to see this tech in action:
# 

# ==> Compare our Imo to MintsHelper <==
Co = scf_wfn.Ca_subset('AO','OCC')
Cv = scf_wfn.Ca_subset('AO','VIR')
MO = np.asarray(mints.mo_eri(Co, Cv, Co, Cv))
print("Do our transformed ERIs match Psi4's? %s" % np.allclose(I_mo, np.asarray(MO)))


# Now we have all the pieces needed to compute $E_0^{(2)}$.  This could be done by writing explicit loops over occupied and virtual indices Python side, e.g.,
# ~~~python
# # Compute SS MP2 Correlation
# mp2_ss_corr = 0.0
# for i in xrange(ndocc):
#     for a in xrange(nmo - ndocc):
#         for j in xrange(ndocc):
#             for b in xrange(nmo - ndocc):
#                 numerator = I_mo[i,a,j,b] * (I_mo[i, a, j, b] - I_mo[i, b, j, a])
#                 mp2_ss_corr += numerator / (e_ij[i] + e_ij[j] - e_ab[a] - e_ab[b])
# ~~~
# This method, while very clear what is going on and easy to program, has the distinct disadvantage that Python loops are much slower than the same block written in a compiled language like C, C++, or Fortran.  For this reason, it is better to use infrastructure available in NumPy like `np.einsum()`, `np.dot`, etc. to explicitly compute the above quantity C-side.  It should be clear how to contract the four-index integrals $(i\,a\mid j\,b)$ and $(i\,a\mid j\,b)$ with one another, but what about the energy eigenvalues $\epsilon$?  We can use a NumPy trick called *broadcasting* to construct a four-index array of all possible energy denominators, which can then be contracted with the full I_mo arrays.  To do this, we'll use the function `np.reshape()`:
# ~~~python
# # Prepare 4d energy denominator array
# e_denom = e_ij.reshape(-1, 1, 1, 1) # Diagonal of 4d array are occupied orbital energies
# e_denom -= e_ab.reshape(-1, 1, 1)   # all combinations of (e_ij - e_ab)
# e_denom += e_ij.reshape(-1, 1)      # all combinations of [(e_ij - e_ab) + e_ij]
# e_denom -= e_ab                     # All combinations of full denominator
# e_denom = 1 / e_denom               # Take reciprocal for contracting with numerator
# ~~~
# In the cell below, compute the energy denominator using `np.reshape()` and contract this array with the four-index ERIs to compute the same-spin and opposite-spin MP2 correction using `np.einsum()`.  Then, add these quantities to the SCF energy computed above to obtain the total MP2 energy.
# 
# Hint: For the opposite-spin correlation, use `np.swapaxes()` to obtain the correct ordering of the indices in the exchange integral.
# 

# ==> Compute MP2 Correlation & MP2 Energy <==
# Compute energy denominator array
e_denom = 1 / (e_ij.reshape(-1, 1, 1, 1) - e_ab.reshape(-1, 1, 1) + e_ij.reshape(-1, 1) - e_ab)

# Compute SS & OS MP2 Correlation with Einsum
mp2_os_corr = np.einsum('iajb,iajb,iajb->', I_mo, I_mo, e_denom)
mp2_ss_corr = np.einsum('iajb,iajb,iajb->', I_mo, I_mo - I_mo.swapaxes(1,3), e_denom)

# Total MP2 Energy
MP2_E = scf_e + mp2_os_corr + mp2_ss_corr


# ==> Compare to Psi4 <==
psi4.driver.p4util.compare_values(psi4.energy('mp2'), MP2_E, 6, 'MP2 Energy')


# ## References
# 
# 1. Original paper: "Note on an Approximation Treatment for Many-Electron Systems"
# 	> [[Moller:1934:618](https://journals.aps.org/pr/abstract/10.1103/PhysRev.46.618)] C. Møller and M. S. Plesset, *Phys. Rev.* **46**, 618 (1934)
# 2. The Laplace-transformation in MP theory: "Minimax approximation for the decomposition of energy denominators in Laplace-transformed Møller–Plesset perturbation theories"
#     > [[Takasuka:2008:044112](http://aip.scitation.org/doi/10.1063/1.2958921)] A. Takatsuka, T. Siichiro, and W. Hackbusch, *J. Phys. Chem.*, **129**, 044112 (2008)
# 3. Equations taken from:
# 	> [[Szabo:1996](https://books.google.com/books?id=KQ3DAgAAQBAJ&printsec=frontcover&dq=szabo+%26+ostlund&hl=en&sa=X&ved=0ahUKEwiYhv6A8YjUAhXLSCYKHdH5AJ4Q6AEIJjAA#v=onepage&q=szabo%20%26%20ostlund&f=false)] A. Szabo and N. S. Ostlund, *Modern Quantum Chemistry: Introduction to Advanced Electronic Structure Theory*. Courier Corporation, 1996.
# 4. Algorithms taken from:
# 	> [Crawford:prog] T. D. Crawford, "The Second-Order Møller–Plesset Perturbation Theory (MP2) Energy."  Accessed via the web at http://sirius.chem.vt.edu/wiki/doku.php?id=crawdad:programming.
# 




# # Direct Inversion of the Iterative Subspace
# 
# When solving systems of linear (or nonlinear) equations, iterative methods are often employed.  Unfortunately, such methods often suffer from convergence issues such as numerical instability, slow convergence, and significant computational expense when applied to difficult problems.  In these cases, convergence accelleration methods may be applied to both speed up, stabilize and/or reduce the cost for the convergence patterns of these methods, so that solving such problems become computationally tractable.  One such method is known as the direct inversion of the iterative subspace (DIIS) method, which is commonly applied to address convergence issues within self consistent field computations in Hartree-Fock theory (and other iterative electronic structure methods).  In this tutorial, we'll introduce the theory of DIIS for a general iterative procedure, before integrating DIIS into our previous implementation of RHF.
# 
# ## I. Theory
# 
# DIIS is a widely applicable convergence acceleration method, which is applicable to numerous problems in linear algebra and the computational sciences, as well as quantum chemistry in particular.  Therefore, we will introduce the theory of this method in the general sense, before seeking to apply it to SCF.  
# 
# Suppose that for a given problem, there exist a set of trial vectors $\{\mid{\bf p}_i\,\rangle\}$ which have been generated iteratively, converging toward the true solution, $\mid{\bf p}^f\,\rangle$.  Then the true solution can be approximately constructed as a linear combination of the trial vectors,
# $$\mid{\bf p}\,\rangle = \sum_ic_i\mid{\bf p}_i\,\rangle,$$
# where we require that the residual vector 
# $$\mid{\bf r}\,\rangle = \sum_ic_i\mid{\bf r}_i\,\rangle\,;\;\;\; \mid{\bf r}_i\,\rangle 
# =\, \mid{\bf p}_{i+1}\,\rangle - \mid{\bf p}_i\,\rangle$$
# is a least-squares approximate to the zero vector, according to the constraint
# $$\sum_i c_i = 1.$$
# This constraint on the expansion coefficients can be seen by noting that each trial function ${\bf p}_i$ may be represented as an error vector applied to the true solution, $\mid{\bf p}^f\,\rangle + \mid{\bf e}_i\,\rangle$.  Then
# \begin{align}
# \mid{\bf p}\,\rangle &= \sum_ic_i\mid{\bf p}_i\,\rangle\&= \sum_i c_i(\mid{\bf p}^f\,\rangle + \mid{\bf e}_i\,\rangle)\&= \mid{\bf p}^f\,\rangle\sum_i c_i + \sum_i c_i\mid{\bf e}_i\,\rangle
# \end{align}
# Convergence results in a minimization of the error (causing the second term to vanish); for the DIIS solution vector $\mid{\bf p}\,\rangle$ and the true solution vector $\mid{\bf p}^f\,\rangle$ to be equal, it must be that $\sum_i c_i = 1$.  We satisfy our condition for the residual vector by minimizing its norm,
# $$\langle\,{\bf r}\mid{\bf r}\,\rangle = \sum_{ij} c_i^* c_j \langle\,{\bf r}_i\mid{\bf r}_j\,\rangle,$$
# using Lagrange's method of undetermined coefficients subject to the constraint on $\{c_i\}$:
# $${\cal L} = {\bf c}^{\dagger}{\bf Bc} - \lambda\left(1 - \sum_i c_i\right)$$
# where $B_{ij} = \langle {\bf r}_i\mid {\bf r}_j\rangle$ is the matrix of residual vector overlaps.  Minimization of the Lagrangian with respect to the coefficient $c_k$ yields (for real values)
# \begin{align}
# \frac{\partial{\cal L}}{\partial c_k} = 0 &= \sum_j c_jB_{jk} + \sum_i c_iB_{ik} - \lambda\&= 2\sum_ic_iB_{ik} - \lambda
# \end{align}
# which has matrix representation
# \begin{equation}
# \begin{pmatrix}
#   B_{11} & B_{12} & \cdots & B_{1m} & -1 \  B_{21} & B_{22} & \cdots & B_{2m} & -1 \  \vdots  & \vdots  & \ddots & \vdots  & \vdots \  B_{n1} & B_{n2} & \cdots & B_{nm} & -1 \  -1 & -1 & \cdots & -1 & 0
# \end{pmatrix}
# \begin{pmatrix}
# c_1\c_2\\vdots \c_n\\lambda
# \end{pmatrix}
# =
# \begin{pmatrix}
# 0\0\\vdots\0\-1
# \end{pmatrix},
# \end{equation}
# 
# which we will refer to as the Pulay equation, named after the inventor of DIIS.  It is worth noting at this point that our trial vectors, residual vectors, and solution vector may in fact be tensors of arbitrary rank; it is for this reason that we have used the generic notation of Dirac in the above discussion to denote the inner product between such objects.
# 
# ## II. Algorithms for DIIS
# The general DIIS procedure, as described above, has the following structure during each iteration:
# #### Algorithm 1: Generic DIIS procedure
# 1. Compute new trial vector, $\mid{\bf p}_{i+1}\,\rangle$, append to list of trial vectors
# 2. Compute new residual vector, $\mid{\bf r}_{i+1}\,\rangle$, append to list of trial vectors
# 3. Check convergence criteria
#     - If RMSD of $\mid{\bf r}_{i+1}\,\rangle$ sufficiently small, and
#     - If change in DIIS solution vector $\mid{\bf p}\,\rangle$ sufficiently small, break
# 4. Build **B** matrix from previous residual vectors
# 5. Solve Pulay equation for coefficients $\{c_i\}$
# 6. Compute DIIS solution vector $\mid{\bf p}\,\rangle$
# 
# For SCF iteration, the most common choice of trial vector is the Fock matrix **F**; this choice has the advantage over other potential choices (e.g., the density matrix **D**) of **F** not being idempotent, so that it may benefit from extrapolation.  The residual vector is commonly chosen to be the orbital gradient in the AO basis,
# $$g_{\mu\nu} = ({\bf FDS} - {\bf SDF})_{\mu\nu},$$
# however the better choice (which we will make in our implementation!) is to orthogonormalize the basis of the gradient with the inverse overlap metric ${\bf A} = {\bf S}^{-1/2}$:
# $$r_{\mu\nu} = ({\bf A}^{\rm T}({\bf FDS} - {\bf SDF}){\bf A})_{\mu\nu}.$$
# Therefore, the SCF-specific DIIS procedure (integrated into the SCF iteration algorithm) will be:
# #### Algorithm 2: DIIS within an SCF Iteration
# 1. Compute **F**, append to list of previous trial vectors
# 2. Compute AO orbital gradient **r**, append to list of previous residual vectors
# 3. Compute RHF energy
# 3. Check convergence criteria
#     - If RMSD of **r** sufficiently small, and
#     - If change in SCF energy sufficiently small, break
# 4. Build **B** matrix from previous AO gradient vectors
# 5. Solve Pulay equation for coefficients $\{c_i\}$
# 6. Compute DIIS solution vector **F_DIIS** from $\{c_i\}$ and previous trial vectors
# 7. Compute new orbital guess with **F_DIIS**
# 

# ## III. Implementation
# 
# In order to implement DIIS, we're going to integrate it into an existing RHF program.  Since we just-so-happened to write such a program in the last tutorial, let's re-use the part of the code before the SCF integration which won't change when we include DIIS:
# 

# ==> Basic Setup <==
# Import statements
import psi4
import numpy as np

# Memory specification
psi4.set_memory(int(5e8))
numpy_memory = 2

# Set output file
psi4.core.set_output_file('output.dat', False)

# Define Physicist's water -- don't forget C1 symmetry!
mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")

# Set computation options
psi4.set_options({'basis': 'cc-pvdz',
                  'scf_type': 'pk',
                  'e_convergence': 1e-8})

# Maximum SCF iterations
MAXITER = 40
# Energy convergence criterion
E_conv = 1.0e-6
D_conv = 1.0e-3


# ==> Static 1e- & 2e- Properties <==
# Class instantiation
wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('basis'))
mints = psi4.core.MintsHelper(wfn.basisset())

# Overlap matrix
S = np.asarray(mints.ao_overlap())

# Number of basis Functions & doubly occupied orbitals
nbf = S.shape[0]
ndocc = wfn.nalpha()

print('Number of occupied orbitals: %d' % ndocc)
print('Number of basis functions: %d' % nbf)

# Memory check for ERI tensor
I_size = (nbf**4) * 8.e-9
print('\nSize of the ERI tensor will be %4.2f GB.' % I_size)
memory_footprint = I_size * 1.5
if I_size > numpy_memory:
    psi4.core.clean()
    raise Exception("Estimated memory utilization (%4.2f GB) exceeds allotted memory                      limit of %4.2f GB." % (memory_footprint, numpy_memory))

# Build ERI Tensor
I = np.asarray(mints.ao_eri())

# Build core Hamiltonian
T = np.asarray(mints.ao_kinetic())
V = np.asarray(mints.ao_potential())
H = T + V


# ==> CORE Guess <==
# AO Orthogonalization Matrix
A = mints.ao_overlap()
A.power(-0.5, 1.e-16)
A = np.asarray(A)

# Transformed Fock matrix
F_p = A.dot(H).dot(A)

# Diagonalize F_p for eigenvalues & eigenvectors with NumPy
e, C_p = np.linalg.eigh(F_p)

# Transform C_p back into AO basis
C = A.dot(C_p)

# Grab occupied orbitals
C_occ = C[:, :ndocc]

# Build density matrix from occupied orbitals
D = np.einsum('pi,qi->pq', C_occ, C_occ)

# Nuclear Repulsion Energy
E_nuc = mol.nuclear_repulsion_energy()


# Now let's put DIIS into action.  Before our iterations begin, we'll need to create empty lists to hold our previous residual vectors (AO orbital gradients) and trial vectors (previous Fock matrices), along with setting starting values for our SCF energy and previous energy:
# 

# ==> Pre-Iteration Setup <==
# SCF & Previous Energy
SCF_E = 0.0
E_old = 0.0


# Now we're ready to write our SCF iterations according to Algorithm 2.  Here are some hints which may help you along the way:
# 
# #### Starting DIIS
# Since DIIS builds the approximate solution vector $\mid{\bf p}\,\rangle$ as a linear combination of the previous trial vectors $\{\mid{\bf p}_i\,\rangle\}$, there's no need to perform DIIS on the first SCF iteration, since there's only one trial vector for DIIS to use!
# 
# #### Building **B**
# 1. The **B** matrix in the Lagrange equation is really $\tilde{\bf B} = \begin{pmatrix} {\bf B} & -1\\ -1 & 0\end{pmatrix}$.
# 2. Since **B** is the matrix of residual overlaps, it will be a square matrix of dimension equal to the number of residual vectors.  If **B** is an $N\times N$ matrix, how big is $\tilde{\bf B}$?
# 3. Since our residuals are real, **B** will be a symmetric matrix.
# 4. To build $\tilde{\bf B}$, make an empty array of the appropriate dimension, then use array indexing to set the values of the elements.
# 
# #### Solving the Pulay equation
# 1. Use built-in NumPy functionality to make your life easier.
# 2. The solution vector for the Pulay equation is $\tilde{\bf c} = \begin{pmatrix} {\bf c}\\ \lambda\end{pmatrix}$, where $\lambda$ is the Lagrange multiplier, and the right hand side is $\begin{pmatrix} {\bf 0}\\ -1\end{pmatrix}$.  
# 

# Start from fresh orbitals
F_p =  A.dot(H).dot(A)
e, C_p = np.linalg.eigh(F_p)
C = A.dot(C_p)
C_occ = C[:, :ndocc]
D = np.einsum('pi,qi->pq', C_occ, C_occ)

# Trial & Residual Vector Lists
F_list = []
DIIS_RESID = []

# ==> SCF Iterations w/ DIIS <==
print('==> Starting SCF Iterations <==\n')

# Begin Iterations
for scf_iter in range(1, MAXITER + 1):
    # Build Fock matrix
    J = np.einsum('pqrs,rs->pq', I, D)
    K = np.einsum('prqs,rs->pq', I, D)
    F = H + 2*J - K
    
    # Build DIIS Residual
    diis_r = A.dot(F.dot(D).dot(S) - S.dot(D).dot(F)).dot(A)
    
    # Append trial & residual vectors to lists
    F_list.append(F)
    DIIS_RESID.append(diis_r)
    
    # Compute RHF energy
    SCF_E = np.einsum('pq,pq->', (H + F), D) + E_nuc
    dE = SCF_E - E_old
    dRMS = np.mean(diis_r**2)**0.5
    print('SCF Iteration %3d: Energy = %4.16f dE = % 1.5E dRMS = %1.5E' % (scf_iter, SCF_E, dE, dRMS))
    
    # SCF Converged?
    if (abs(dE) < E_conv) and (dRMS < D_conv):
        break
    E_old = SCF_E
    
    if scf_iter >= 2:
        # Build B matrix
        B_dim = len(F_list) + 1
        B = np.empty((B_dim, B_dim))
        B[-1, :] = -1
        B[:, -1] = -1
        B[-1, -1] = 0
        for i in range(len(F_list)):
            for j in range(len(F_list)):
                B[i, j] = np.einsum('ij,ij->', DIIS_RESID[i], DIIS_RESID[j])

        # Build RHS of Pulay equation 
        rhs = np.zeros((B_dim))
        rhs[-1] = -1
        
        # Solve Pulay equation for c_i's with NumPy
        coeff = np.linalg.solve(B, rhs)
        
        # Build DIIS Fock matrix
        F = np.zeros_like(F)
        for x in range(coeff.shape[0] - 1):
            F += coeff[x] * F_list[x]
    
    # Compute new orbital guess with DIIS Fock matrix
    F_p =  A.dot(F).dot(A)
    e, C_p = np.linalg.eigh(F_p)
    C = A.dot(C_p)
    C_occ = C[:, :ndocc]
    D = np.einsum('pi,qi->pq', C_occ, C_occ)
    
    # MAXITER exceeded?
    if (scf_iter == MAXITER):
        psi4.core.clean()
        raise Exception("Maximum number of SCF iterations exceeded.")

# Post iterations
print('\nSCF converged.')
print('Final RHF Energy: %.8f [Eh]' % SCF_E)


# Congratulations! You've written your very own Restricted Hartree-Fock program with DIIS convergence accelleration!  Finally, let's check your final RHF energy against <span style='font-variant: small-caps'> Psi4</span>:
# 

# Compare to Psi4
SCF_E_psi = psi4.energy('SCF')
psi4.driver.p4util.compare_values(SCF_E_psi, SCF_E, 6, 'SCF Energy')


# ## References
# 1. P. Pulay. *Chem. Phys. Lett.* **73**, 393-398 (1980)
# 2. C. David Sherrill. *"Some comments on accellerating convergence of iterative sequences using direct inversion of the iterative subspace (DIIS)".* Available at: vergil.chemistry.gatech.edu/notes/diis/diis.pdf. (1998)
# 




# # Density Fitting
# 
# Density fitting is an extremely useful tool to reduce the computational scaling of many quantum chemical methods.  
# Density fitting works by approximating the four-index electron repulsion integral (ERI) tensors from Hartree-Fock 
# theory, $g_{\mu\nu\lambda\sigma} = (\mu\nu|\lambda\sigma)$, by
# 
# $$(\mu\nu|\lambda\sigma) \approx \widetilde{(\mu\nu|P)}[J^{-1}]_{PQ}\widetilde{(Q|\lambda\sigma)}$$
# 
# where the Coulomb metric $[J]_{PQ}$ and the three-index integral $\widetilde{(Q|\lambda\sigma)}$ are defined as
# 
# \begin{align}
# [J]_{PQ} &= \int P(r_1)\frac{1}{r_{12}}Q(r_2){\rm d}^3r_1{\rm d}^3r_2\\widetilde{(Q|\lambda\sigma)} &= \int Q(r_1)\frac{1}{r_{12}}\lambda(r_2)\sigma(r_2){\rm d}^3r_1{\rm d}^3r_2
# \end{align}
# 
# To simplify the density fitting notation, the inverse Coulomb metric is typically folded into the three-index tensor:
# 
# \begin{align}
# (P|\lambda\sigma) &= [J^{-\frac{1}{2}}]_{PQ}\widetilde{(Q|\lambda\sigma)}\g_{\mu\nu\lambda\sigma} &\approx (\mu\nu|P)(P|\lambda\sigma)
# \end{align}
# 
# These transformed three-index tensors can then be used to compute various quantities, including the four-index ERIs, 
# as well as Coulomb (J) and exchange (K) matrices, and therefore the Fock matrix (F).  Before we go any further, let's
# see how to generate these transformed tensors using <span style='font-variant: small-caps'> Psi4</span>.  
# 
# First, let's import <span style='font-variant: small-caps'> Psi4</span> and set up some global options, as well as
# define a molecule and initial wavefunction:
# 

# ==> Psi4 & NumPy options, Geometry Definition <==
import numpy as np
import psi4

# Set numpy defaults
np.set_printoptions(precision=5, linewidth=200, suppress=True)

# Set Psi4 memory & output options
psi4.set_memory(int(2e9))
psi4.core.set_output_file('output.dat', False)

# Geometry specification
mol = psi4.geometry("""
O
H 1 0.96
H 1 0.96 2 104.5
symmetry c1
""")

# Psi4 options
psi4.set_options({'basis': 'aug-cc-pvdz',
                  'scf_type': 'df',
                  'e_convergence': 1e-10,
                  'd_convergence': 1e-10})

wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('basis'))


# ## Building the Auxiliary Basis Set
# 
# One component of our density-fitted tensors $g_{\mu\nu\lambda\sigma} \approx (\mu\nu|P)(P|\lambda\sigma)$ which
# is unique from their exact, canonical counterparts $(\mu\nu|\lambda\sigma)$ is the additional "auxiliary" index, $P$.
# This index corresponds to inserting a resolution of the identity, which is expanded in an auxiliary basis set $\{P\}$.
# In order to build our density-fitted integrals, we first need to generate this auxiliary basis set.  Fortunately,
# we can do this with the `psi4.core.BasisSet` object:
# ~~~python
# # Build auxiliary basis set
# aux = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", "", "JKFIT", "aug-cc-pVDZ")
# ~~~
# 
# There are special fitting basis sets that are optimal for a given orbital basis. As we will be building J and K 
# objects we want the `JKFIT` basis associated with the orbital `aug-cc-pVDZ` basis. This basis is straightfowardly 
# named `aug-cc-pVDZ-jkfit`.
# 

# Build auxiliary basis set
aux = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", "", "JKFIT", "aug-cc-pVDZ")


# ## Building Density-Fitted ERIs
# Now, we can use our orbital and auxiliary basis sets to construct the `Qpq` object with the inverted metric. As the 
# tensors are very similar to full ERI's we can use the same computer for both with the aid of a "zero basis". If we 
# think carefully about the $\widetilde{(Q|\lambda\sigma)}$ and $(\mu\nu|\lambda\sigma)$ we should note that on the 
# right and left hand sides the two gaussian basis functions are contracted to a single density.
# 
# Specifically, for $\widetilde{(Q|\lambda\sigma)}$ the right hand side is a single basis function without being 
# multiplied by another, so we can "trick" the MintsHelper object into computing these quanties if we have a "basis 
# set" which effectively does not act on another. This is, effectively, what a "zero basis" does.
# 
# The $[J^{-\frac{1}{2}}]_{PQ}$ object can be built in a similar way where we use the Psi4 Matrix's built in `power` 
# function to raise this to the $-\frac{1}{2}$ power. The call `Matrix.power(-0.5, 1.e-14)` will invert the Matrix to 
# the $-\frac{1}{2}$ while guarding against values smaller than 1.e-14. Recall that machine epsilon is ~1.e-16, when 
# these small values are taken to a negative fractional power they could become very large and dominate the resulting 
# matrix even though they are effectively noise before the inversion.
# 
# ~~~python
# orb = wfn.basisset()
# zero_bas = psi4.core.BasisSet.zero_ao_basis_set()
# 
# # Build MintsHelper Instance
# mints = psi4.core.MintsHelper(orb)
# 
# # Build (P|pq) raw 3-index ERIs, dimension (1, Naux, nbf, nbf)
# Ppq = mints.ao_eri(zero_bas, aux, orb, orb)
# 
# # Build and invert the metric
# metric = psi4.core.ao_eri(zero_bas, aux, zero_bas, aux)
# metric.power(-0.5, 1.e-14)
# 
# # Remove the excess dimensions of Ppq & metric
# Ppq = np.squeeze(Ppq)
# metric = np.squeeze(metric)
# 
# # Contract Ppq & Metric to build Qso
# Qso = np.einsum('QP,Ppq->Qpq', metric, Ppq)
# ~~~
# 

# ==> Build Density-Fitted Integrals <==
# Get orbital basis & build zero basis
orb = wfn.basisset()
zero_bas = psi4.core.BasisSet.zero_ao_basis_set()

# Build instance of MintsHelper
mints = psi4.core.MintsHelper(orb)

# Build (P|pq) raw 3-index ERIs, dimension (1, Naux, nbf, nbf)
Ppq = mints.ao_eri(zero_bas, aux, orb, orb)

# Build & invert Coulomb metric, dimension (1, Naux, 1, Naux)
metric = mints.ao_eri(zero_bas, aux, zero_bas, aux)
metric.power(-0.5, 1.e-14)

# Remove excess dimensions of Ppq, & metric
Ppq = np.squeeze(Ppq)
metric = np.squeeze(metric)

# Build the Qso object
Qpq = np.einsum('QP,Ppq->Qpq', metric, Ppq)


# ## Example: Building a Density-Fitted Fock Matrix
# Now that we've obtained our `Qpq` tensors, we may use them to build the Fock matrix.  To do so, since we aren't 
# implementing a fully density-fitted RHF program, we'll first need to get a density matrix and one-electron Hamiltonian 
# from somewhere. Let's get them from a converged HF wavefunction, so we can check our work later:
# 

# ==> Compute SCF Wavefunction, Density Matrix, & 1-electron H <==
scf_e, scf_wfn = psi4.energy('scf', return_wfn=True)
D = scf_wfn.Da()
H = scf_wfn.H()


# Now that we have our density-fitted integrals and a density matrix, we can build a Fock matrix.  There are several 
# different algorithms which we can successfully use to do so; for now, we'll use a simple algorithm and `np.einsum()` 
# to illustrate how to perform contractions with these density fitted tensors and leave a detailed discussion of those 
# algorithms/different tensor contraction methods elsewhere.  Recall that the Fock matrix, $F$, is given by
# 
# $$F = H + 2J - K,$$
# 
# where $H$ is the one-electron Hamiltonian matrix, $J$ is the Coulomb matrix, and $K$ is the exchange matrix.  The 
# Coulomb and Exchange matrices have elements guven by
# 
# \begin{align}
# J[D_{\lambda\sigma}]_{\mu\nu} &= (\mu\nu|\lambda\sigma)D_{\lambda\sigma}\K[D_{\lambda\sigma}]_{\mu\nu} &= (\mu\lambda|\nu\sigma)D_{\lambda\sigma}.
# \end{align}
# 
# When employing conventional 4-index ERI tensors, computing both $J$ and $K$ involves contracting over four unique
# indices, which involves four distinct loops -- one over each unique index in the contraction.  Therefore, the 
# scaling of this procedure is $\mathcal{O}(N^4)$, where $N$ is the number of iterations in each loop (one for each 
# basis function).  The above expressions can be coded using `np.einsum()` to handle the tensor contractions:
# 
# ~~~python
# J = np.einsum('pqrs,rs->pq', I_pqrs, D)
# K = np.einsum('prqs,rs->pq', I_pqrs, D)
# ~~~
# 
# for exact ERIs `I_pqrs`.  If we employ density fitting, however, we can reduce this scaling by reducing the number 
# of unique indices involved in the contractions.  Substituting in the density-fitted $(P|\lambda\sigma)$ tensors into 
# the above expressions, we obtain the following:
# 
# \begin{align}
# J[D_{\lambda\sigma}]_{\mu\nu} &= (\mu\nu|P)(P|\lambda\sigma)D_{\lambda\sigma}\K[D_{\lambda\sigma}]_{\mu\nu} &= (\mu\lambda|P)(P|\nu\sigma)D_{\lambda\sigma}.
# \end{align}
# 
# Naively, this seems like we have actually *increased* the scaling of our algorithm, because we have added the $P$ 
# index to the expression, bringing the total to five unique indices, meaning this would scale like .  We've actually 
# made our lives easier, however: with three different tensors to contract, we can perform one contraction at a time!  
# 
# For $J$, this works out to the following two-step procedure:
# 
# \begin{align}
# \chi_P &= (P|\lambda\sigma)D_{\lambda\sigma} \J[D_{\lambda\sigma}]_{\mu\nu} &= (\mu\nu|P)\chi_P
# \end{align}
# 
# 
# In the cell below, using `np.einsum()` and our `Qpq` tensor, try to construct `J`:
# 

# Two-step build of J with Qpq and D
X_Q = np.einsum('Qpq,pq->Q', Qpq, D)
J = np.einsum('Qpq,Q->pq', Qpq, X_Q)


# Each of the above contractions, first constructing the `X_Q` intermediate and finally the full Coulomb matrix `J`, only involve three unique indices.  Therefore, the Coulomb matrix build above scales as $\mathcal{O}(N_{\rm aux}N^2)$.  Notice that we have distinguished the number of auxiliary ($N_{\rm aux}$) and orbital ($N$) basis functions; this is because auxiliary basis sets are usually around double the size of their corresponding orbital counterparts.  
# 
# We can play the same intermediate trick for building the Exchange matrix $K$:
# 
# \begin{align}
# \zeta_{P\nu\lambda} &= (P|\nu\sigma)D_{\lambda\sigma} \K[D_{\lambda\sigma}]_{\mu\nu} &= (\mu\lambda|P)\zeta_{P\nu\lambda}
# \end{align}
# 
# Just like with $J$, try building $K$ in the cell below:
# 

# Two-step build of K with Qpq and D
Z_Qqr = np.einsum('Qrs,sq->Qrq', Qpq, D)
K = np.einsum('Qpq,Qrq->pr', Qpq, Z_Qqr)


# Unfortunately, our two-step $K$ build does not incur a reduction in the overall scaling of the algorithm, with each contraction above scaling as $\mathcal{O}(N^3N_{\rm aux})$. The major benefit of density fitting for $K$ builds comes in the form of the small storage overhead of the three-index `Qpq` tensors compared to the full four-index `I_pqrs` tensors.  Even when exploiting the full eight-fold symmetry of the $(\mu\nu|\lambda\sigma)$ integrals, storing `I_pqrs` for a system with 3000 AO basis functions will require 81 TB of space, compared to a mere 216 GB to store the full `Qpq` object when exploiting the twofold symmetry of $(P|\lambda\sigma)$.  
# 
# Now that we've built density-fitted versions of the $J$ and $K$ matrices, let's check our work by comparing a Fock matrix built using our $J$ and $K$ with the fully converged Fock matrix from our original SCF/aug-cc-pVDZ computation.  
# 
# Below, build F using the one-electron Hamiltonian from the converged SCF wavefuntion and our $J$ and $K$ matrices.  Then, get the converged $F$ from the SCF wavefunction:
# 

# Build F from SCF 1 e- Hamiltonian and our density-fitted J & K
F = H + 2 * J - K
# Get converged Fock matrix from converged SCF wavefunction
scf_F = scf_wfn.Fa()


# Feeling lucky? Execute the next cell to see if you've computed $J$, $K$, and $F$ correctly:
# 

if np.allclose(F, scf_F):
    print("Nicely done!! Your density-fitted Fock matrix matches Psi4!")
else:
    print("Whoops...something went wrong.  Try again!")


# Finally, we can remember the identity of the $D$ matrix for SCF which will be $D_{\lambda\sigma} = C_{\lambda i} C_{\sigma i}$, where $i$ is the occupied index. We can factor our $K$ build once more:
# \begin{align}
# D_{\lambda\sigma} &= C_{\lambda i} C_{\sigma i} \\
# \zeta_{P\nu i} &= (P|\nu\sigma)C_{\sigma i} \\
# K[D_{\lambda\sigma}]_{\mu\nu} &= \zeta_{P\nu i}\zeta_{P\mu i}
# \end{align}
# 
# Consider the ratio between the number of basis functions and the size of the occupied index. Why would the above be beneficial?




# ## References
# 1. F. Weigend, Phys. Chem. Chem. Phys. 4, 4285 (2002).
# 2. O. Vahtras, J. Alml ̈of, and M. W. Feyereisen, Chem. Phys. Lett. 213, 514 (1993).
# 3. B. I. Dunlap, J. W. D. Connolly, and J. R. Sabin, J. Chem. Phys. 71, 3396 (1979).
# 4. J. L. Whitten, J. Chem. Phys. 58, 4496 (1973).
# 




# # Molecule Objects in <span style="font-variant: small-caps"> Psi4 </span>
# 
# This tutorial provides an overview on creating and manipulating molecule objects in <span style='font-variant: small-caps'> Psi4</span>, illustrated with an example parameterization of the Lennard-Jones potential for Helium dimer.  
# 
# In order to use <span style="font-variant: small-caps"> Psi4 </span> within a Python environment, we may import <span style="font-variant: small-caps"> Psi4 </span> just as any other module: 
# 
# (note: the `PYTHONPATH` environment variable must be set correctly, check [here](https://github.com/dgasmith/psi4numpy/blob/master/README.md) for more details)
# 

import psi4


# Unlike in <span style="font-variant: small-caps"> Psi4 </span> input files, defining a molecule in Python is done by passing the molecular coordinates as a triple-quoted string to the [`psi4.geometry()`](http://psicode.org/psi4manual/master/api/psi4.driver.geometry.html#psi4.driver.geometry "API Details") function:
# 

he = psi4.geometry("""
He
""")


# Here, not only does the variable `he` refer to the helium molecule, but also an instance of the [`psi4.core.Molecule`](http://psicode.org/psi4manual/master/psi4api.html#psi4.core.Molecule "Go to API")
# class in <span style='font-variant: small-caps'> Psi4</span>; this will be discussed in more detail later.  For a more
# complicated system than an isolated atom, the coordinates can be given in Cartesian or Z-Matrix formats:
# 

h2o = psi4.geometry("""
O
H 1 0.96
H 1 0.96 2 104.5
""")


# Information like the molecular charge, multiplicity, and units are assumed to be 0, 1, and Angstroms, respectively, if not specified within the molecule definition.  This is done by adding one or more [molecule keywords](http://psicode.org/psi4manual/master/psithonmol.html#molecule-keywords "Go to Documentation") to the geometry string used to define the molecule.  Additionally, <span style="font-variant: small-caps"> Psi4 </span> can detect molecular symmetry, or it can be specified manually.  For example, to define a doublet water cation in $C_{2V}$ symmetry using lengths in Bohr,
# 

doublet_h2o_cation = psi4.geometry("""
1 2
O
H 1 1.814
H 1 1.814 2 104.5

units bohr
symmetry c2v
""")


# where the line `1 2` defines the charge and multiplicity, respectively.  For systems of non-bonded fragments, the coordinates of each fragment are separated by a double-hyphen `"--"`; this allows for one fragment to be defined with Cartesian and another to be defined with Z-Matrix. For example, the hydronium-benzene complex can be defined with: 
# 

hydronium_benzene = psi4.geometry("""
0 1
C          0.710500000000    -0.794637665924    -1.230622098778
C          1.421000000000    -0.794637665924     0.000000000000
C          0.710500000000    -0.794637665924     1.230622098778
C         -0.710500000000    -0.794637665924     1.230622098778
H          1.254500000000    -0.794637665924    -2.172857738095
H         -1.254500000000    -0.794637665924     2.172857738095
C         -0.710500000000    -0.794637665924    -1.230622098778
C         -1.421000000000    -0.794637665924     0.000000000000
H          2.509000000000    -0.794637665924     0.000000000000
H          1.254500000000    -0.794637665924     2.172857738095
H         -1.254500000000    -0.794637665924    -2.172857738095
H         -2.509000000000    -0.794637665924     0.000000000000
-- 
1 1
X  1  CC  3  30   2  A2
O  13 R   1  90   2  90
H  14 OH  13 TDA  1  0
H  14 OH  15 TDA  13 A1
H  14 OH  15 TDA  13 -A1

CC    = 1.421
CH    = 1.088
A1    = 120.0
A2    = 180.0
OH    = 1.05
R     = 4.0
units angstrom
""")


# For non-bonded fragments, the charge and multiplicity should be given explicitly for each fragment.  If not, the 
# charge and multiplicity given (or inferred) for the first fragment is assumed to be the same for all fragments.  In 
# addition to defining the coordinates outright, we have used variables within the geometry specification strings to
# define bond lengths, angles, and dihedrals in the molecule.  Similarly, we could define the X, Y, or Z Cartesian
# coordinate for any atom in our molecule. 
# 
# In order to define these variables after the molecule is built, as opposed to within the geometry specification 
# itself, there are several ways to do so; one of which will be illustrated in the Lennard-Jones potential example 
# below. 
# 
# When a Psi4 molecule is first built using ``psi4.geometry()``, it is in an unfinished state, as a user may wish to 
# tweak the molecule. This can be solved by calling [``psi4.Molecule.update_geometry()``](http://psicode.org/psi4manual/master/psi4api.html#psi4.core.Molecule.update_geometry "Go to API"). This will update the molecule and restore sanity 
# to chemistry.  
# 

h2cch2 = psi4.geometry("""
H
C 1 HC
H 2 HC 1 A1
C 2 CC 3 A1 1 D1
H 4 HC 2 A1 1 D1
H 4 HC 2 A1 1 D2

HC = 1.08
CC = 1.4
A1 = 120.0
D1 = 180.0
D2 = 0.0
""")

print("Ethene has %d atoms" % h2cch2.natom())


h2cch2.update_geometry()
print("Ethene has %d atoms" % h2cch2.natom())


# Finally, one can obtain useful information from a molecule by invoking one of several [`psi4.core.Molecule`](http://psicode.org/psi4manual/master/psi4api.html#psi4.core.Molecule "Go to Documentation") class methods on the molecule of interest.  For example, if we were interested in verifying that our doublet water cation from above is, in fact, a doublet, we could invoke
# ~~~python
# doublet_h2o_cation.multiplicity()
# ~~~
# Below, some useful Molecule class methods are tabulated; please refer to the documentation for more details.
# 
# | Method | Description |
# |--------|-------------|
# | [center_of_mass()](http://psicode.org/psi4manual/master/psi4api.html#psi4.core.Molecule.center_of_mass "Go to Documentation") | Computes center of mass of molecule |
# | [molecular_charge()](http://psicode.org/psi4manual/master/psi4api.html#psi4.core.Molecule.molecular_charge "Go to Documentation") | Gets the molecular charge |
# | [multiplicity()](http://psicode.org/psi4manual/master/psi4api.html#psi4.core.Molecule.multiplicity "Go to Documentation") | Gets the total multiplicity | 
# | [nuclear_repulsion_energy()](http://psicode.org/psi4manual/master/psi4api.html#psi4.core.Molecule.nuclear_repulsion_energy "Go to Documentation") | Computes the nuclear repulsion energy of the molecule |
# 

# ## Example: Fitting Lennard-Jones Parameters from Potential Energy Scan
# 
# In this example, we will compute and fit a potential energy curve for the Helium dimer.  To begin with, let's create a string representation for our He dimer in Z-Matrix format, with the variable `**R**` representing the distance between the He atoms. The stars surrounding the ``R`` aren't any special syntax, just a convenient marker for future substitution.
# 

# Define He Dimer
he_dimer = """
He
--
He 1 **R**
"""


# Now we can build a series of dimers with the He atoms at different separations, and compute the energy at each point:
# 

distances = [2.875, 3.0, 3.125, 3.25, 3.375, 3.5, 3.75, 4.0, 4.5, 5.0, 6.0, 7.0]
energies = []
for d in distances:
    # Build a new molecule at each separation
    mol = psi4.geometry(he_dimer.replace('**R**', str(d)))
    
    # Compute the Counterpoise-Corrected interaction energy
    en = psi4.energy('MP2/aug-cc-pVDZ', molecule=mol, bsse_type='cp')

    # Place in a reasonable unit, Wavenumbers in this case
    en *= 219474.6
    
    # Append the value to our list
    energies.append(en)

print("Finished computing the potential!")


# Next, we can use the [NumPy](http://www.numpy.org/) library to fit a curve to these points along the potential scan.  In this case, we will fit a Lennard-Jones potential.
# 

import numpy as np


# Fit data in least-squares way to a -12, -6 polynomial
powers = [-12, -6]
x = np.power(np.array(distances).reshape(-1, 1), powers)
coeffs = np.linalg.lstsq(x, energies)[0]

# Build list of points
fpoints = np.linspace(2, 7, 50).reshape(-1, 1)
fdata = np.power(fpoints, powers)

fit_energies = np.dot(fdata, coeffs)


# To visualize our results, we can use the [Matplotlib](http://matplotlib.org/) library.  Since we're working in a Jupyter notebook, we can also use the `%matplotlib inline` "magic" command so that the graphs will show up in the notebook itself (check [here](https://ipython.org/ipython-doc/3/interactive/magics.html) for a comprehensive list of magic commands).
# 

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


plt.xlim((2, 7))  # X limits
plt.ylim((-7, 2))  # Y limits
plt.scatter(distances, energies)  # Scatter plot of the distances/energies
plt.plot(fpoints, fit_energies)  # Fit data
plt.plot([0,10], [0,0], 'k-')  # Make a line at 0





# # <span style="font-family: Optima, sans-serif; color: #273896;">P<span style="font-size: 82%;">SI</span>4</span> Acquisition
# 

# ### Easy choice: clone and compile
# 
# * Clone from [GitHub](https://github.com/psi4/psi4)
# * Compile via [CMake](https://github.com/psi4/psi4/blob/master/CMakeLists.txt#L16-L123)
# * Build FAQ and guide [here](http://psicode.org/psi4manual/master/build_faq.html)
# 

# ### Easier choice: conda
# 
# 0. Get [miniconda](https://conda.io/miniconda.html) or [anaconda](https://www.continuum.io/downloads) installer or script
# 0. Run the installer or "bash" the script on the command line, accepting its license and allowing it to add its installation directory to your `PATH` environment variable: _e.g._, ``bash Anaconda3-4.3.0-Linux-x86_64.sh``
# 0. Create a <span style="font-family: Optima, sans-serif; color: #273896;">P<span style="font-size: 82%;">SI</span>4</span> environment named "p4env". Until the 1.1 release, instructions may be found [at the end of this thread](https://github.com/psi4/psi4/issues/466#issuecomment-272589229)
# 0. Activate the environment: ``source activate p4env``
# 0. See [guide](http://psicode.org/psi4manual/master/build_planning.html#faq-runordinarymodule) for any configuration trouble
# 

# ### Easiest choice: conda installer
# 
# * **Not available until 1.1 release**
# * ~~Get Psi4+miniconda installer script from [psicode](http://psicode.org/downloads.html) and follow its directions~~
# 

# ### Online choice: binder button
# * **Not available until 1.1 release**
# 

# # <span style="font-family: Optima, sans-serif; color: #273896;">P<span style="font-size: 82%;">SI</span>4</span> Boilerplate
# 
# 

# ### Eponymous Python module imports
# 

import psi4
import numpy as np


# ### Direct output and scratch
# 
# * Output goes to file ``output.dat``
# * Boolean directs overwriting (`True`) rather than appending (`False`).
# * Optionally, redirect scratch away from ``/tmp`` to existing, writable directory
# 

psi4.set_output_file("output.dat", True)

# optional
psi4.core.IOManager.shared_object().set_default_path("/scratch")


# ### Set memory limits
# 
# * Give 500 Mb of memory to <span style="font-family: Optima, sans-serif; color: #273896;">P<span style="font-size: 82%;">SI</span>4</span>
# * Give 2 Gb of memory for NumPy arrays (quantity for Psi4NumPy project, *not* passed to NumPy)
# * Sum of these two should nowhere approach the RAM of your computer
# 

psi4.set_memory(int(5e8))
numpy_memory = 2


# ### Molecule and Basis
# 
# * Covered in detail in subsequent tutorials. This is the quick reference
# * Running _without_ symmetry recommended in Psi4NumPy for simplest expressions
# 

psi4.geometry("""
O 0.0 0.0 0.0 
H 1.0 0.0 0.0
H 0.0 1.0 0.0
symmetry c1
""")

psi4.set_options({'basis': 'cc-pvdz'})


"""Tutorial: Describing the implementation of density-fitted MP2 from an RHF reference"""

__author__    = "Dominic A. Sirianni"
__credit__    = ["Dominic A. Sirianni", "Daniel G. A. Smith"]

__copyright__ = "(c) 2014-2017, The Psi4NumPy Developers"
__license__   = "BSD-3-Clause"
__date__      = "2017-05-24"


# # Density Fitted MP2
# 
# As we saw in tutorial (5a), the single most expensive step for a conventional MP2 program using full ERIs is the integral transformation from the atomic orbital (AO) to molecular orbital (MO) basis, scaling as ${\cal O}(N^5)$.  The scaling of this step may be reduced to ${\cal O}(N^4)$ if we employ density fitting, as the three-index density fitted tensors may be transformed individually into the MO basis before being recombined to form the full four-index tensors in the MO basis needed by the MP2 energy expression.  This tutorial will discuss the specific challenges encountered when applying density fitting to an MP2 program.
# 
# ### Implementation
# The first part of our DF-MP2 program will look exactly the same as the conventional MP2 program that we wrote in (5a), with the exception that we must specify the `scf_type df` and omit the option `mp2_type conv` within the `psi4.set_options()` block, to ensure that we are employing density fitting in the Hartree-Fock reference.  Below, implement the following:
# 
# - Import Psi4 and NumPy, and set memory & output file
# - Define our molecule and Psi4 options
# - Compute the RHF reference wavefucntion and energy
# - Obtain the number of occupied and virtual MOs, and total number of MOs
# - Get the orbital energies and coefficient matrix; partition into occupied & virtual blocks
# 

# ==> Import statements & Global Options <==
import psi4
import numpy as np

psi4.set_memory(int(2e9))
numpy_memory = 2
psi4.core.set_output_file('output.dat', False)


# ==> Options Definitions & SCF E, Wfn <==
mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")


psi4.set_options({'basis': 'aug-cc-pvdz',
                  'scf_type': 'df',
                  'e_convergence': 1e-8,
                  'd_convergence': 1e-8})

# Get the SCF wavefunction & energies
scf_e, scf_wfn = psi4.energy('scf', return_wfn=True)

# Number of Occupied orbitals & MOs
ndocc = scf_wfn.nalpha()
nmo = scf_wfn.nmo()
nvirt = nmo - ndocc

# Get orbital energies, cast into NumPy array, and separate occupied & virtual
eps = np.asarray(scf_wfn.epsilon_a())
e_ij = eps[:ndocc]
e_ab = eps[ndocc:]

# Get MO coefficients from SCF wavefunction
C = np.asarray(scf_wfn.Ca())
Cocc = C[:, :ndocc]
Cvirt = C[:, ndocc:]


# From the conventional MP2 program, we know that the next step is to obtain the ERIs and transform them into the MO basis using the orbital coefficient matrix, **C**.  In order to do this using density-fitted integrals, must first build and transform the DF-ERI's similar to that in the density-fitted HF chapter.
# 

# ==> Density Fitted ERIs <==
# Build auxiliar basis set
aux = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", "", "JKFIT", "aug-cc-pVDZ")

# Build instance of Mints object
orb = scf_wfn.basisset()
mints = psi4.core.MintsHelper(orb)

# Build a zero basis
zero_bas = psi4.core.BasisSet.zero_ao_basis_set()

# Raw 3-index
Ppq = np.squeeze(mints.ao_eri(zero_bas, aux, orb, orb))

# Build and invert the Coulomb metric
metric = mints.ao_eri(zero_bas, aux, zero_bas, aux)
metric.power(-0.5, 1.e-14)
metric = np.squeeze(metric)

Qpq = np.einsum("QP,Ppq->Qpq", metric, Ppq)


# Now that we have our three-index integrals, we are able to transform them into the MO basis.  To do this, we could simply use `np.einsum()` to carry out the transformation in a single step:
# ~~~python
# # Transform Qpq -> Qmo @ O(N^5)
# Qmo = np.einsum('pi,Qpq,qj->Qij', C, Qpq, C)
# ~~~
# This simple transformation works, but it doesn't reduce the caling of the transformation.  This approach saves over the conventional one only because a single ${\cal O}(N^5)$ transformation would need to be done, instead of four.  We can, however, borrow the idea from conventional MP2 to carry out the transformation in more than one step, saving the intermediates along the way.  Using this approach, we are able to transform the `Qpq` tensors into the MO basis in two successive ${\cal O}(N^4)$ steps.  In the cell below, transform the `Qpq` tensors with this reduced scaling algorithm, and save the occupied-virtual slice of the full `Qmo` tensor:
# 

# ==> Transform Qpq -> Qmo @ O(N^4) <==
Qmo = np.einsum('pi,Qpq->Qiq', C, Qpq)
Qmo = np.einsum('Qiq,qj->Qij', Qmo, C)

# Get Occupied-Virtual Block
Qmo = Qmo[:, :ndocc, ndocc:]


# We are now ready to compute the DF-MP2 correlation energy $E_0^{(2)}$.  One approach for doing this would clearly be to form the four-index OVOV $(ia\mid jb)$ ERI tensor directly [an ${\cal O}(N^5)$ contraction], and proceed exactly as we did for conventional MP2.  This would, however, result in needing to store this entire tensor in memory, which would be prohibitive for large systems/basis sets and would only result in minimal savings.  A more clever (and much less memory-intensive) algorithm can be found by considering the MP2 correlation energy expressions,
# 
# \begin{equation}
# E_{\rm 0,\,SS}^{(2)} = \sum_{ij}\sum_{ab}\frac{(ia\mid jb)[(ia\mid jb) - (ib\mid ja)]}{\epsilon_i - \epsilon_a + \epsilon_j - \epsilon_b},\,{\rm and}
# \end{equation}
# \begin{equation}
# E_{\rm 0,\,OS}^{(2)} = \sum_{ij}\sum_{ab}\frac{(ia\mid jb)(ia\mid jb)}{\epsilon_i - \epsilon_a + \epsilon_j - \epsilon_b},
# \end{equation}
# 
# for particular values of the occupied orbital indices $i$ and $j$:
# 
# \begin{equation}
# E_{\rm 0,\,SS}^{(2)}(i, j) = \sum_{ab}\frac{I_{ab}[I_{ab} - I_{ba}]}{\epsilon_i + \epsilon_j - \boldsymbol{\epsilon}_{ab}}
# \end{equation}
# \begin{equation}
# E_{\rm 0,\,OS}^{(2)}(i, j) = \sum_{ab}\frac{I_{ab}I_{ab}}{\epsilon_i + \epsilon_j - \boldsymbol{\epsilon}_{ab}},
# \end{equation}
# 
# for virtual-virtual blocks of the full ERI tensors $I_{ab}$ and a matrix $\boldsymbol{\epsilon}_{ab}$ containing all possible combinations of the virtual orbital energies $\epsilon_a$ and $\epsilon_b$.  These expressions are advantageous because they only call for two-index contractions between the virtual-virtual blocks of the OVOV ERI tensor, and the storage of only the VV-block of this tensor in memory.  Furthermore, the formation of the $I_{ab}$ tensor is also ameliorated, since only the auxiliary-virtual blocks of the three-index `Qmo` tensor must be contracted, which can be done on-the-fly as opposed to beforehand (requiring no storage in memory).  In practice, these expressions can be used within explicit loops over occupied indices $i$ and $j$; therefore the overall scaling of this step is still ${\cal O}(N^5)$ (formation of $I_{ab}$ is ${\cal O}(N^3)$ inside two loops), however the the drastically reduced memory requirements result in this method a significant win over conventional MP2.
# 
# One potentially mysterious quantity in the frozen-index expressions given above is the virtual-virtual orbital eigenvalue tensor, **$\epsilon$**.  To build this array, we can again borrow an idea from our implementation of conventional MP2: reshaping and broadcasting.  In the cell below, use these techniques to build the VV $\boldsymbol{\epsilon}_{ab}$ tensor.
# 
# Hint: In the frozen-index expressions above, $\boldsymbol{\epsilon}_{ab}$ is *subtracted* from the occupied orbital energies $\epsilon_i$ and $\epsilon_j$.  Therefore, the virtual orbital energies should be added together to have the correct sign!
# 

# ==> Build VV Epsilon Tensor <==
e_vv = e_ab.reshape(-1, 1) + e_ab


# In addition to the memory savings incurred by generating VV-blocks of our ERI tensors on-the-fly, we can exploit the permutational symmetry of these tensors [Sherrill:ERI] to drastically reduce the number of loops (and therefore Qv,Qv contractions!) which are needed to compute the MP2 correlation energy.  To see the relevant symmetry, recall that a spin-free four index ERI over spatial orbitals (written in chemists' notation) is given by
# 
# $$(i\,a\mid j\,b) = \int{\rm d}^3{\bf r}_1{\rm d}^3{\bf r}_2\phi_i^*({\bf x}_1)\phi_a({\bf x}_1)\frac{1}{r_{12}}\phi_j^*({\bf x}_2)\phi_b({\bf x}_2)$$
# 
# For real orbitals, it is easy to see that $(i\,a\mid j\,b) = (j\,b\mid i\,a)$; therefore, it is unnecessary to iterate over all combinations of $i$ and $j$, since the value of the contractions containing either $(i\,a\mid j\,b)$ or $(j\,b\mid i\,a)$ will be identical.  Therefore, it suffices to iterate over all $i$ and only $j\geq i$.  Then, the "diagonal elements" ($i = j$) will contribute once to each of the same-spin and opposite-spin correlation energies, and the "off-diagonal" elements ($i\neq j$) will contribute twice to each correlation energy due to symmetry.  This corresponds to placing either a 1 or a 2 in the numerator of the energy denominator, i.e., 
# 
# \begin{equation}
# E_{denom} = \frac{\alpha}{\epsilon_i + \epsilon_j - \boldsymbol{\epsilon}_{ab}};\;\;\;\alpha = \begin{cases}1;\; i=j\\2;\;i\neq j\end{cases},
# \end{equation}
# 
# before contracting this tensor with $I_{ab}$ and $I_{ba}$ to compute the correlation energy.  In the cell below, compute the same-spin and opposite-spin DF-MP2 correlation energies using the frozen-index expressions 3 and 4 above, exploiting the permutational symmetry of the full $(ia\mid jb)$ ERIs.  Then, using the correlation energies, compute the total MP2 energy using the DF-RHF energy we computed above.
# 

mp2_os_corr = 0.0
mp2_ss_corr = 0.0
for i in range(ndocc):
    # Get epsilon_i from e_ij
    e_i = e_ij[i]
    
    # Get 2d array Qa for i from Qov
    i_Qa = Qmo[:, i, :]
    
    for j in range(i, ndocc):
        # Get epsilon_j from e_ij
        e_j = e_ij[j]
        
        # Get 2d array Qb for j from Qov
        j_Qb = Qmo[:, j, :]
        
        # Compute 2d ERI array for fixed i,j from Qa & Qb
        ij_Iab = np.einsum('Qa,Qb->ab', i_Qa, j_Qb)

        # Compute energy denominator
        if i == j:
            e_denom = 1.0 / (e_i + e_j - e_vv)
        else:
            e_denom = 2.0 / (e_i + e_j - e_vv)

        # Compute SS & OS MP2 Correlation
        mp2_os_corr += np.einsum('ab,ab,ab->', ij_Iab, ij_Iab, e_denom)
        mp2_ss_corr += np.einsum('ab,ab,ab->', ij_Iab, ij_Iab - ij_Iab.T, e_denom)

# Compute MP2 correlation & total MP2 Energy
mp2_corr = mp2_os_corr + mp2_ss_corr
MP2_E = scf_e + mp2_corr


# ==> Compare to Psi4 <==
psi4.driver.p4util.compare_values(psi4.energy('mp2'), MP2_E, 6, 'MP2 Energy')


# ## References
# 
# 1. Original paper: "Note on an Approximation Treatment for Many-Electron Systems"
# 	> [[Moller:1934:618](https://journals.aps.org/pr/abstract/10.1103/PhysRev.46.618)] C. Møller and M. S. Plesset, *Phys. Rev.* **46**, 618 (1934)
# 2. The Laplace-transformation in MP theory: "Minimax approximation for the decomposition of energy denominators in Laplace-transformed Møller–Plesset perturbation theories"
#     > [[Takasuka:2008:044112](http://aip.scitation.org/doi/10.1063/1.2958921)] A. Takatsuka, T. Siichiro, and W. Hackbusch, *J. Phys. Chem.*, **129**, 044112 (2008)
# 3. Equations taken from:
# 	> [[Szabo:1996](https://books.google.com/books?id=KQ3DAgAAQBAJ&printsec=frontcover&dq=szabo+%26+ostlund&hl=en&sa=X&ved=0ahUKEwiYhv6A8YjUAhXLSCYKHdH5AJ4Q6AEIJjAA#v=onepage&q=szabo%20%26%20ostlund&f=false)] A. Szabo and N. S. Ostlund, *Modern Quantum Chemistry: Introduction to Advanced Electronic Structure Theory*. Courier Corporation, 1996.
# 4. Algorithms taken from:
# 	> [Crawford:prog] T. D. Crawford, "The Second-Order Møller–Plesset Perturbation Theory (MP2) Energy."  Accessed via the web at http://sirius.chem.vt.edu/wiki/doku.php?id=crawdad:programming.
# 5. ERI Permutational Symmetries
# 	> [Sherrill:ERI] C. David Sherrill, "Permutational Symmetries of One- and Two-Electron Integrals." Accessed via the web at http://vergil.chemistry.gatech.edu/notes/permsymm/permsymm.pdf.
# 




# # VV10 Non-local correlation kernel
# One of the largest deficiencies with semilocal functionals is the lack of long-range correlation effects. This most notable expresses itself as the lack of disperssion in the interactions between molecules. VV10 was expressly created to bridge the gap between the expensive of true non-local correlation and a computational tractable form. We will begin by writing the overall expression:
# 
# $$E_c^{\rm{nl}} = \frac{1}{2}\int \int d{\bf r}d{\bf r'}\rho({\bf r})\Phi({\bf r},{\bf r'})\rho({\bf r'})$$
# 
# Where the two densities are tied together through the $\Phi$ operator.
# 
# For VV10 we have:
# $$
# \begin{align}
# \Phi &= -\frac{3}{2gg'(g + g')}\ g &= \omega_0({\rm r}) R^2 + \kappa({\rm r)}\ g' &= \omega_0({\rm r}) R^2 + \kappa({\rm r')}
# \end{align}
# $$
# 
# Where $w_{0}$:
# 
# $$
# \begin{align}
# \omega_{0}(r) &= \sqrt{\omega_{g}^2(r) + \frac{\omega_p^2(r)}{3}} \\omega_g^2(r) &= C \left | \frac{\nabla \rho({\bf r})}{\rho({\bf r})} \right |^4 \\omega_p^2(r) &= 4 \pi \rho({\bf r})
# \end{align}
# $$
# 
# and finally:
# 
# $$\kappa({\bf r}) = b * \frac{3 \pi}{2} \left [ \frac{\rho({\bf r})}{9\pi} \right ]^\frac{1}{6}$$
# 
# While there are several expression, this is actually quite easy to compute. First let us examine how the VV10 energy is reintegrated:
# 
# $$E_c^{\rm{VV10}} = \int d{\bf r} \rho{\bf r} \left [ \beta + \frac{1}{2}\int d{\bf r'} \rho{\bf r'} \Phi({\bf r},{\bf r'}) \right]$$
# 
# 

import psi4
import numpy as np
import ks_helper as ks

mol = psi4.geometry("""
He 0 0 -5
He 0 0  5
symmetry c1
""")
options = {'BASIS':               'aug-cc-pVDZ',
           'DFT_SPHERICAL_POINTS': 110,
           'DFT_RADIAL_POINTS':    20}


# ## VV10 coefficients
# First let us build set and build a few static coefficients:
# 

coef_C = 0.0093
coef_B = 5.9
coef_beta = 1.0 / 32.0 * (3.0 / (coef_B ** 2.0)) ** (3.0 / 4.0)


# ## VV10 kernel
# First let us construct a function that compute $\omega_0$ and $\kappa$ quantities. To make one piece simpler let us first examine a piece of $\omega_g$:
# $$\left |\frac{\nabla \rho({\bf r})}{\rho({\bf r})} \right|^4$$
# 
# quantity. Recall that 
# 
# $$\gamma({\bf r}) = \nabla\rho({\bf r})\cdot\nabla\rho({\bf r})$$
# 
# therefore, we can simplify the above to:
# 
# $$\frac{\nabla \rho({\bf r})}{\rho({\bf r})} = \left | \frac{\gamma({\bf r})}{\rho({\bf r})\cdot({\bf r})} \right | ^2 $$
# 

def compute_vv10_kernel(rho, gamma):
    kappa_pref = coef_B * (1.5 * np.pi) / ((9.0 * np.pi) ** (1.0 / 6.0))
    
    # Compute R quantities
    Wp = (4.0 / 3.0) * np.pi * rho
    Wg = coef_C * ((gamma / (rho * rho)) ** 2.0)
    W0 = np.sqrt(Wg + Wp)
    
    kappa = rho ** (1.0 / 6.0) * kappa_pref
    return W0, kappa


# ## VV10 energy and gradient evaluation
# 
# Yup so just integrate it out. Pretty easy.
# 

def compute_vv10(D, Vpot):


    nbf = D.shape[0]
    Varr = np.zeros((nbf, nbf))
    
    total_e = 0.0
    tD = 2.0 * np.array(D)
    
    points_func = Vpot.properties()[0]
    superfunc = Vpot.functional()

    xc_e = 0.0
    vv10_e = 0.0
    
    # First loop over the outer set of blocks
    for l_block in range(Vpot.nblocks()):
        
        # Obtain general grid information
        l_grid = Vpot.get_block(l_block)
        l_w = np.array(l_grid.w())
        l_x = np.array(l_grid.x())
        l_y = np.array(l_grid.y())
        l_z = np.array(l_grid.z())
        l_npoints = l_w.shape[0]

        points_func.compute_points(l_grid)

        
        # Compute the functional itself
        ret = superfunc.compute_functional(points_func.point_values(), -1)
        
        xc_e += np.vdot(l_w, np.array(ret["V"])[:l_npoints])
        v_rho = np.array(ret["V_RHO_A"])[:l_npoints]
        v_gamma = np.array(ret["V_GAMMA_AA"])[:l_npoints]
        
        # Begin VV10 information
        l_rho = np.array(points_func.point_values()["RHO_A"])[:l_npoints]
        l_gamma = np.array(points_func.point_values()["GAMMA_AA"])[:l_npoints]
        
        l_W0, l_kappa = compute_vv10_kernel(l_rho, l_gamma)
        
        phi_kernel = np.zeros_like(l_rho)
        phi_U = np.zeros_like(l_rho)
        phi_W = np.zeros_like(l_rho)
        
        # Loop over the inner set of blocks
        for r_block in range(Vpot.nblocks()):
            
            # Repeat as for the left blocks
            r_grid = Vpot.get_block(r_block)
            r_w = np.array(r_grid.w())
            r_x = np.array(r_grid.x())
            r_y = np.array(r_grid.y())
            r_z = np.array(r_grid.z())
            r_npoints = r_w.shape[0]

            points_func.compute_points(r_grid)

            r_rho = np.array(points_func.point_values()["RHO_A"])[:r_npoints]
            r_gamma = np.array(points_func.point_values()["GAMMA_AA"])[:r_npoints]
        
            r_W0, r_kappa = compute_vv10_kernel(r_rho, r_gamma)
            
            # Build the distnace matrix
            R2  = (l_x[:, None] - r_x) ** 2
            R2 += (l_y[:, None] - r_y) ** 2
            R2 += (l_z[:, None] - r_z) ** 2
            
            # Build g
            g = l_W0[:, None] * R2 + l_kappa[:, None]
            gp = r_W0 * R2 + r_kappa
        
            # 
            F_kernal = -1.5 * r_w * r_rho / (g * gp * (g + gp))
            F_U = F_kernal * ((1.0 / g) + (1.0 / (g + gp)))
            F_W = F_U * R2


            phi_kernel += np.sum(F_kernal, axis=1)
            phi_U += -np.sum(F_U, axis=1)
            phi_W += -np.sum(F_W, axis=1)
            
        # Compute those derivatives
        kappa_dn = l_kappa / (6.0 * l_rho)
        w0_dgamma = coef_C * l_gamma / (l_W0 * l_rho ** 4.0)
        w0_drho = 2.0 / l_W0 * (np.pi/3.0 - coef_C * np.power(l_gamma, 2.0) / (l_rho ** 5.0))

        # Sum up the energy
        vv10_e += np.sum(l_w * l_rho * (coef_beta + 0.5 * phi_kernel))

        # Perturb the derivative quantities
        v_rho += coef_beta + phi_kernel + l_rho * (kappa_dn * phi_U + w0_drho * phi_W)
        v_rho *= 0.5
        
        v_gamma += l_rho * w0_dgamma * phi_W

        # Recompute to l_grid
        lpos = np.array(l_grid.functions_local_to_global())
        points_func.compute_points(l_grid)
        nfunctions = lpos.shape[0]
        
        # Integrate the LDA and GGA quantities
        phi = np.array(points_func.basis_values()["PHI"])[:l_npoints, :nfunctions]
        phi_x = np.array(points_func.basis_values()["PHI_X"])[:l_npoints, :nfunctions]
        phi_y = np.array(points_func.basis_values()["PHI_Y"])[:l_npoints, :nfunctions]
        phi_z = np.array(points_func.basis_values()["PHI_Z"])[:l_npoints, :nfunctions]
        
        # LDA
        Vtmp = np.einsum('pb,p,p,pa->ab', phi, v_rho, l_w, phi)

        # GGA
        l_rho_x = np.array(points_func.point_values()["RHO_AX"])[:l_npoints]
        l_rho_y = np.array(points_func.point_values()["RHO_AY"])[:l_npoints]
        l_rho_z = np.array(points_func.point_values()["RHO_AZ"])[:l_npoints]
        
        tmp_grid = 2.0 * l_w * v_gamma
        Vtmp += np.einsum('pb,p,p,pa->ab', phi, tmp_grid, l_rho_x, phi_x)
        Vtmp += np.einsum('pb,p,p,pa->ab', phi, tmp_grid, l_rho_y, phi_y)
        Vtmp += np.einsum('pb,p,p,pa->ab', phi, tmp_grid, l_rho_z, phi_z)
        
        # Sum back to the correct place
        Varr[(lpos[:, None], lpos)] += Vtmp + Vtmp.T
        
    print("   VV10 NL energy: %16.8f" % vv10_e)
        
    xc_e += vv10_e
    return xc_e, Varr

ks.ks_solver("VV10", mol, options, compute_vv10)       


# Refs:
#  - Vydrov O. A.; Van Voorhis T., *J. Chem. Phys.*, **2010**, *133*, 244103
# 

