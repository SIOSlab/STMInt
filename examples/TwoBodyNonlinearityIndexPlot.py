from sympy import *
import numpy as np
from STMint.STMint import STMint
import math
import matplotlib.pyplot as plt

# Calculating the solutions to non-Dimensional two-body motion
integ = STMint(preset="twoBody", variational_order=2)

[states,STMs,STTs,ts] = integ.dynVar_int2([0,(20.*math.pi)],[1,0,0,0,1,0], output='all', max_step=.1)

NLIs = []
norms = []
for i in range(len(ts)):
	NLIs.append(integ.nonlin_index(STMs[i], STTs[i]))
	norms.append(np.linalg.norm(STMs[i]))


# Plotting
xvals = []

for i in range(0,21,2):
    xvals.append(math.pi*i)

xlabels = [0]
for i in range(2,21,2):
    xlabels.append(str(i) + r'$\pi$')

plt.figure(figsize=(8,6))
plt.plot(ts,NLIs)
plt.title("Nonlinearity Associated With Non-dimensional Circular Two-Body Motion")
plt.xlabel("Time")
plt.xticks(xvals,xlabels)
plt.ylabel("Nonlinearity Index")

plt.figure(figsize=(8,6))
plt.plot(ts,norms)
plt.title("Nonlinearity Associated With Non-dimensional Circular Two-Body Motion")
plt.xlabel("Time")
plt.xticks(xvals,xlabels)
plt.ylabel("Nonlinearity Index")

plt.figure(figsize=(8,6))
plt.plot(np.array(states)[:,0],np.array(states)[:,1])
plt.title("state plot")
plt.xlabel("x")
plt.ylabel("y")
plt.show()