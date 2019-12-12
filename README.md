# TOY EXPERIMENT

Goal is to understand *Auto-Encoding Variational Bayes* through a 
toy experiment. 

In this experiment, the data is drawn from the following process:

1. z is drawn from a gaussian with mean 0 and variance 100.
2. x is drawn from a gaussian with mean [z, z] and a |z| std dev in
dimension.
