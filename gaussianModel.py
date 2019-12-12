# Train Gaussian VAE
# python3 gaussianModel.py <fileName> <epochs>
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt

def klDivergence(logvar, mean) : 
    var = torch.exp(logvar) 
    meanSq = mean * mean
    kl = -0.5 * torch.sum(1 + logvar - meanSq - var, dim=1) 
    kl = kl.unsqueeze(1)
    return kl

def logGaussianPDF(logvar, mean, x) :
    diff1 = (x[:, 0] - mean[:,0])**2
    diff2 = (x[:, 1] - mean[:,1])**2
    
    var = torch.exp(logvar)

    term1 = -0.5 * ((diff1 / var[:,0]) + (diff2 / var[:,1]))
    term2 = -math.log(2 * math.pi) - 0.5 * (logvar[:,0] + logvar[:,1])

    return term1 + term2

class ApproxPosteriorTransformation (nn.Module) :
    """ 
    This network represents g_{\Phi}, the differentiable
    tranformation used in the reparamtrization trick.
    """

    def __init__ (self, h) :
        super(ApproxPosteriorTransformation, self).__init__()
        self.layer1 = torch.nn.Linear(2, h)

        self.logvar = torch.nn.Linear(h, 1)
        self.mean = torch.nn.Linear(h, 1)

    def forward (self, x, e) : 
        h = torch.tanh(self.layer1(x))

        logvar = self.logvar(h)
        std = torch.sqrt(torch.exp(logvar))

        mean = self.mean(h)

        z = mean + (e * std)

        kl = klDivergence(logvar, mean)
        return torch.cat([kl, z], dim=1)

class ApproxConditionalDistribution (nn.Module) :
    """
    This networks represents P(x^{(i)}| z). 
    The conditional probability distribution. 
    """

    def __init__ (self, h) :
        super(ApproxConditionalDistribution, self).__init__()
        # Choosing diagonal covariance matrix for convenience.
        self.layer1 = torch.nn.Linear(1, h)

        self.logvar = torch.nn.Linear (h, 2)
        self.mean = torch.nn.Linear(h, 2)
    
    def forward (self, z) : 
        h = torch.tanh(self.layer1(z))
        logvar = self.logvar(h)
        mean = self.mean(h)
        gaussParams = torch.cat([logvar, mean], dim=1)
        return gaussParams

def main () :
    fname = sys.argv[1]
    epochs = int(sys.argv[2])

    q = ApproxPosteriorTransformation(5)
    p = ApproxConditionalDistribution(5)

    qOpt = torch.optim.Adam(
            q.parameters(), 
            lr=4e-2,
            weight_decay=1e-6)
    pOpt = torch.optim.Adam(
            p.parameters(), 
            lr=4e-2,
            weight_decay=1e-6)

    data = torch.tensor(np.loadtxt(fname)).float()
    N = data.shape[0]

    reconLoss = []
    klLoss = []
    lbLoss = []

    for epoch in range(epochs) :
        qOpt.zero_grad()
        pOpt.zero_grad()

        eps = torch.randn(N, 1)
        o = q(data, eps)
        z = o[:,1]
        z = z.unsqueeze(1)

        v1 = p(z)

        logvar = v1[:,0:2]
        mean = v1[:,2:]

        term1 = torch.sum(logGaussianPDF(logvar, mean, data))
        kl = torch.sum(o[:,0])

        lb = term1 - kl
        lb_ = -lb
        lb_.backward()
        
        qOpt.step()
        pOpt.step()

        print("Epoch: ", epoch, ", Variational LB: ", lb.item(), ", KL: ", kl.item(), ", Recon: ", term1.item())

        reconLoss.append(term1.item())
        klLoss.append(kl.item())
        lbLoss.append(lb.item())


    plt.plot(list(range(epochs)), reconLoss, c='b', label="Conditional")
    plt.plot(list(range(epochs)), klLoss, c='r', label="KL")
    plt.plot(list(range(epochs)), lbLoss, c='g', label="Total")

    plt.xlabel("Epochs")
    plt.ylabel("Variational Lower Bound")
    plt.legend(loc="lower right")
    plt.show()

    torch.save(p.state_dict(), 'p.pkl')
    torch.save(q.state_dict(), 'q.pkl')
    
if __name__ == "__main__"  :
    main()

