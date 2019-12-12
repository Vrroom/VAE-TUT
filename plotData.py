# Plot data from file
# python3 plotData.py <fileName>
import sys
import numpy as np
import matplotlib.pyplot as plt

fname = sys.argv[1]
x = np.loadtxt(fname)
plt.scatter(x[:,0], x[:,1])
plt.show()
