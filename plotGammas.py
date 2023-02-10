###################################################################################################################
#This script plots the gamma distribution from which we sample our epsilons
#
###################################################################################################################
import numpy as np
import matplotlib.pyplot as plt

###################################################################################################################




#parameters
shape = 0.25
scale = 0.025
numberOfSamples = 10000

#sample the distribution
gammaSamples = np.random.gamma(shape=shape, scale=scale, size=numberOfSamples)

#plot a histogram
ourHist = plt.hist(gammaSamples, 100, density=True)
plt.xlabel('$\epsilon$')
plt.ylabel('frequency')
plt.show()