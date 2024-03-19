import qupled.quantum as qpq

# Define a Qstls object to solve the QSTLS scheme
qvsstls = qpq.QVSStls(10.0,
                    1.0,
                    mixing = 0.5,
                    resolution = 0.1,
                    cutoff = 10,
                    matsubara = 16,
                    threads = 1)

# Solve the QSTLS scheme and store the internal energy (v1 calculation)
qvsstls.compute()
uInt1 = qvsstls.computeInternalEnergy()

# Pass in input the fixed component of the auxiliary density response
qvsstls.inputs.fixed = "adr_fixed_rs10.000_theta1.000_QSTLS.bin"

# Repeat the calculation and recompute the internal energy (v2 calculation)
qvsstls.compute()
uInt2 = qvsstls.computeInternalEnergy()

# Compare the internal energies obtained with the two methods
print("Internal energy (v1) = %.8f" % uInt1)
print("Internal energy (v2) = %.8f" % uInt2)

# Change the coupling parameter
qvsstls.inputs.coupling = 20.0

# Compute with the updated coupling parameter
qvsstls.compute()

# Change the degeneracy parameter
qvsstls.inputs.degeneracy = 2.0

# Compute with the update degeneracy parameter (this throws an error)
qvsstls.compute() 
