import numpy as np
from STMint.STMint import STMint
from STMint import TensorNormUtilities as tnu
import math
import matplotlib.pyplot as plt
import numpy.linalg

# Calculating the solutions to non-dimensional two-body motion
integ = STMint(preset="twoBody", variational_order=2)

#integrate variational equations
#10 periods of a circular orbit with unit radius
[states,STMs,STTs,ts] = integ.dynVar_int2([0,(20.*math.pi)],[1,0,0,0,1,0], output='all', max_step=.1)

#calculate nonlinearity index from the STMs and STTs over time
NLI1s = []
NLI2s = []
NLI3s = []
NLI4s = []
NLI5s = []
NLI6s = []
tensNorms = []
norms = []
for i in range(len(ts)):
    NLI1s.append(tnu.nonlin_index_inf_2(STMs[i], STTs[i]))
    NLI2s.append(tnu.nonlin_index_unfold(STMs[i], STTs[i]))
    NLI3s.append(tnu.nonlin_index_2(STMs[i], STTs[i]))
    NLI4s.append(tnu.nonlin_index_frob(STMs[i], STTs[i]))
    NLI5s.append(tnu.nonlin_index_2_eigenvector(STMs[i], STTs[i]))
    NLI6s.append(tnu.nonlin_index_2_eigenvector_symmetrizing(STMs[i], STTs[i]))
    tensNorms.append(tnu.stt_2_norm(STMs[i], STTs[i]))
    norms.append(np.linalg.norm(STMs[i]))


# Plotting
xvals = []

for i in range(0,21,2):
    xvals.append(math.pi*i)

xlabels = [0]
for i in range(2,21,2):
    xlabels.append(str(i) + r'$\pi$')

plt.figure(figsize=(8,6))
#plt.plot(ts,np.array(NLI2s)-np.array(NLI3s))
#plt.plot(ts,np.array(NLI4s)-np.array(NLI3s))
#plt.plot(ts,np.array(NLI5s)-np.array(NLI3s))
plt.plot(ts,np.array(NLI5s)-np.array(NLI6s))
plt.figure(figsize=(8,6))
plt.title("2nd Order STT Norm Associated With Non-dimensional Circular Two-Body Motion")
plt.xlabel("Time")
plt.xticks(xvals,xlabels)
plt.ylabel("Tensor 2-Norm")
plt.plot(ts,tensNorms)

plt.figure(figsize=(8,6))
plt.plot(ts,NLI1s)
plt.plot(ts,NLI2s)
plt.plot(ts,NLI3s)
plt.plot(ts,NLI4s)
plt.plot(ts,NLI5s)
plt.title("Nonlinearity Associated With Non-dimensional Circular Two-Body Motion")
plt.xlabel("Time")
plt.xticks(xvals,xlabels)
plt.ylabel("Nonlinearity Index")

plt.show()
