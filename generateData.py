# Generate data as specified in README.md
# python3 generateData.py <fileName> <numberOfPoints> <seed>
import sys
import numpy as np

fname = sys.argv[1]
nPoints = int(sys.argv[2])

# z = np.random.randn(nPoints)
# x1 = z + np.random.randn(nPoints) * ((np.abs(z) > 1.5) * z * 0.5)
# x2 = z + np.random.randn(nPoints) * ((np.abs(z) > 1.5) * z * 0.5)
# x = np.vstack([x1, x2]).T
# 
# np.savetxt(fname, x)

z = np.random.randint(0, 2, nPoints)

nZeros = np.sum(z == 0)
nOnes = np.sum(z == 1)

x1 = np.zeros(nPoints)
x2 = np.zeros(nPoints)

x1[z == 0] = 1 + np.random.randn(nZeros) * .1
x2[z == 0] = 1 + np.random.randn(nZeros) * .1

x1[z == 1] = -1 + np.random.randn(nOnes) * .1
x2[z == 1] = -1 + np.random.randn(nOnes) * .1

x = np.vstack([x1, x2]).T
np.savetxt(fname, x)

