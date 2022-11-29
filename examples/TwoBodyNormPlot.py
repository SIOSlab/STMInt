from sympy import *
import numpy as np
from STMint.STMint import STMint
import math
import matplotlib.pyplot as plt

# Calculating the Soltuions to Non-Dimensional Two-Body Motion
example = STMint(preset="twoBody")

exampleSol = example.dynVar_int([0,(20*math.pi)],[1,0,0,0,1,0], max_step=.1)

# Isolating the State Transition Matricies
flatSTMs = []

for i in range(len(exampleSol.y[0])):
    stm = []

    for j in range(6,len(exampleSol.y)):
        stm.append(exampleSol.y[j][i])

    flatSTMs.append(stm)

# Calculating the Frobenius Norms of each STM
norms = []

for flatSTM in flatSTMs:
    stm = np.reshape(flatSTM, (6,6))

    norms.append(np.linalg.norm(stm))

# Creating x-values and x-labels for x-axis of graph
xvals = []

for i in range(0,21,2):
    xvals.append(math.pi*i)

xlabels = [0]
for i in range(2,21,2):
    xlabels.append(str(i) + r'$\pi$')

# Plotting
plt.figure(figsize=(8,6))
plt.plot(exampleSol.t,norms)
plt.title("Norm of State Transition Matrix Associated With Non-dimensional Two-Body Motion")
plt.xlabel("Time (periods)")
plt.xticks(xvals,xlabels)
plt.ylabel("Norm")
plt.show()
