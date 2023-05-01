import numpy as np
import qupled.Static as Static

# Define a Qstls object to solve the QSTLS scheme
qstls = Static.Qstls(10.0, 1.0,
                     mixing = 0.5,
                     resolution = 0.1,
                     cutoff = 10,
                     matsubara = 16,
                     threads = 16)

# Solve the QSTLS scheme
qstls.compute()

# Plot the ideal and auxiliary density response for a few matsubara frequencies
qstls.plot(["idr", "adr"], matsubara = np.arange(1, 10, 2))

# Define a QstlsIet object to solve one of the QSTLS-IET schemes
qstls = Static.QstlsIet(30.0, 1.0,
                        "QSTLS-LCT"
                        mixing = 0.2,
                        resolution = 0.1,
                        cutoff = 10,
                        matsubara = 16,
                        scheme2DIntegrals = "segregated"
                        threads = 16)

# solve the QSTLS-IET scheme
qstls.compute()