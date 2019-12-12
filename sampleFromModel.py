# Sample x from model.
# python3 sampleFromModel <pModel> <nPoints>
import sys
from gaussianModel import ApproxConditionalDistribution
import torch
import matplotlib.pyplot as plt
import numpy as np

pModel = sys.argv[1]
nPoints = int(sys.argv[2])

p = ApproxConditionalDistribution(5)

p.load_state_dict(torch.load(pModel))

z = torch.randn(nPoints, 1)
params = p(z)

logvar = params[:,0:2].detach().numpy()
mean = params[:,2:].detach().numpy()

muX = mean[:,0]
muY = mean[:,1]

stdX = np.sqrt(np.exp(logvar[:,0]))
stdY = np.sqrt(np.exp(logvar[:,1]))

X = muX + stdX * np.random.randn(nPoints)
Y = muY + stdY * np.random.randn(nPoints)

plt.scatter(X, Y)
plt.show()
