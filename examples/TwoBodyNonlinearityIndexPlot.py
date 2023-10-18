import numpy as np
from STMint.STMint import STMint
from STMint import TensorNormUtilities as tnu
import math
import matplotlib.pyplot as plt
import numpy.linalg

# Calculating the solutions to non-dimensional two-body motion
integ = STMint(preset="twoBody", variational_order=2)

# integrate variational equations
# 10 periods of a circular orbit with unit radius
states, stms, stts, ts = integ.dynVar_int2(
    [0, (20.0 * math.pi)], [1, 0, 0, 0, 1, 0], output="all", max_step=0.1
)

# calculate nonlinearity index from the stms and stts over time
NLI1s = []
NLI2s = []
NLI3s = []
NLI4s = []
NLI5s = []
NLI6s = []
tensNorms = []
norms = []

# Eliminating first element to not divide by zero in following functions
ts = ts[1:]
for i in range(len(ts)):
    i = i + 1
    NLI1s.append(tnu.nonlin_index_inf_2(stms[i], stts[i]))
    NLI2s.append(tnu.nonlin_index_unfold(stms[i], stts[i]))
    NLI3s.append(tnu.nonlin_index_2(stms[i], stts[i]))
    NLI4s.append(tnu.nonlin_index_frob(stms[i], stts[i]))
    NLI5s.append(tnu.nonlin_index_2_eigenvector(stms[i], stts[i]))
    NLI6s.append(tnu.nonlin_index_2_eigenvector_symmetrizing(stms[i], stts[i]))
    _, norm = tnu.stt_2_norm(stms[i], stts[i])
    tensNorms.append(norm)
    norms.append(np.linalg.norm(stms[i]))

# Plotting
xvals = []
for i in range(0, 21, 2):
    xvals.append(math.pi * i)

xlabels = [0]
for i in range(2, 21, 2):
    xlabels.append(str(i) + r"$\pi$")

plt.figure(figsize=(8, 6))
# plt.plot(ts,np.array(NLI2s)-np.array(NLI3s))
# plt.plot(ts,np.array(NLI4s)-np.array(NLI3s))
# plt.plot(ts,np.array(NLI5s)-np.array(NLI3s))
plt.plot(ts, np.array(NLI5s) - np.array(NLI6s))
plt.figure(figsize=(8, 6))
plt.title("2nd Order STT Norm Associated With Non-dimensional Circular Two-Body Motion")
plt.xlabel("Time")
plt.xticks(xvals, xlabels)
plt.ylabel("Tensor 2-Norm")
plt.plot(ts, tensNorms)

plt.figure(figsize=(8, 6))
plt.plot(ts, NLI1s)
plt.plot(ts, NLI2s)
plt.plot(ts, NLI3s)
plt.plot(ts, NLI4s)
plt.plot(ts, NLI5s)
plt.title("Nonlinearity Associated With Non-dimensional Circular Two-Body Motion")
plt.xlabel("Time")
plt.xticks(xvals, xlabels)
plt.ylabel("Nonlinearity Index")

plt.show()
