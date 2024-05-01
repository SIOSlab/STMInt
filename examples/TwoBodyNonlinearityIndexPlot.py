import numpy as np
from STMint.STMint import STMint
from STMint import TensorNormUtilities as tnu
import math
import matplotlib.pyplot as plt
import numpy.linalg

# Calculating the solutions to non-dimensional two-body motion
integ = STMint(preset="twoBody", variational_order=2)

# integrate variational equations
states, stms, stts, ts = integ.dynVar_int2(
    [0, (2.0 * math.pi)], [1, 0, 0, 0, 1, 0], output="all", max_step=0.01
)

# calculate nonlinearity index from the stms and stts over time
NLI1s = []
NLI2s = []
NLI3s = []
NLI4s = []
NLI5s = []
NLI6s = []

# Eliminating first element to not divide by zero in following functions
ts = ts[1:]
for i in range(len(ts)):
    i = i + 1
    NLI1s.append(tnu.nonlin_index_unfold_bound(stms[i], stts[i]))
    NLI2s.append(tnu.nonlin_index_inf_2(stms[i], stts[i]))
    NLI3s.append(tnu.nonlin_index_2(stms[i], stts[i]))
    NLI4s.append(tnu.nonlin_index_junkins_scale_free(stms[i], stts[i]))
    NLI5s.append(tnu.nonlin_index_DEMoN2(stms[i], stts[i]))
    NLI6s.append(np.sqrt(tnu.nonlin_index_TEMoN3(stms[i], stts[i])))

xvals = np.array([0, 0.25, 0.5, 0.75, 1]) * 2.0 * np.pi
xlabels = ["0", "1/4", "1/2", "3/4", "1"]

# # Plotting
# xvals = []
# for i in range(0, 5, 2):
#     xvals.append(math.pi * i)

# xlabels = [0]
# for i in range(2, 5, 2):
#     xlabels.append(str(i) + r"$\pi$")

plt.style.use("seaborn-v0_8-colorblind")

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(ts, NLI1s, label="2-Norm Bound", linewidth=4)
ax.plot(ts, NLI2s, label="(\u221e, 2)-Norm", linewidth=4)
ax.plot(ts, NLI3s, label="2-Norm", linewidth=4)
ax.plot(ts, NLI4s, label="(Frobenius, 2)-Norm", linewidth=4)
# ax.set_title("Nonlinearity Associated With Non-dimensional Circular Two-Body Motion")
ax.set_xlabel("Time (Periods)", fontsize=18)
ax.set_xticks(xvals, xlabels, fontsize=16)
ax.tick_params(axis="y", labelsize=16)
ax.set_ylabel("Nonlinearity Index", fontsize=18)
ax.legend(fontsize=12)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(ts, NLI1s, label="2-Norm Bound", linewidth=4)
ax.plot(ts, NLI2s, label="(\u221e, 2)-Norm", linewidth=4)
ax.plot(ts, NLI3s, label="2-Norm", linewidth=4)
ax.plot(ts, NLI4s, label="(Frobenius, 2)-Norm", linewidth=4)
ax.plot(ts, NLI6s, label="TEMoN-3", linewidth=4)
# ax.set_title("Nonlinearity Associated With Non-dimensional Circular Two-Body Motion")
ax.set_xlabel("Time (Periods)", fontsize=18)
ax.set_xticks(xvals, xlabels, fontsize=16)
ax.tick_params(axis="y", labelsize=16)
ax.set_ylabel("Nonlinearity Index", fontsize=18)
ax.legend(fontsize=12)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(
    ts, np.array(NLI1s) - np.array(NLI3s), label="2-Norm Bound - 2-Norm", linewidth=4
)
ax.plot(
    ts,
    np.array(NLI4s) - np.array(NLI3s),
    label="(Frobenius, 2)-Norm - 2-Norm",
    linewidth=4,
)
ax.legend(fontsize=12)
ax.set_xlabel("Time (Periods)", fontsize=18)
ax.set_xticks(xvals, xlabels, fontsize=16)
ax.tick_params(axis="y", labelsize=16)
ax.set_ylabel("Nonlinearity Index", fontsize=18)

# fig, ax = plt.subplots(figsize=(8, 6))
# ax.semilogy(ts, NLI5s)
# ax.set_xlabel("Time (Periods)",fontsize=18)
# ax.set_xticks(xvals, xlabels, fontsize=16)
# ax.tick_params(axis='y', labelsize=16)
# ax.set_ylabel("Nonlinearity Index", fontsize=18)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(ts, NLI5s, linewidth=4)
ax.set_xlabel("Time (Periods)", fontsize=18)
ax.set_xticks(xvals, xlabels, fontsize=16)
ax.tick_params(axis="y", labelsize=16)
ax.set_ylabel("DEMoN-2", fontsize=18)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(ts, NLI6s, linewidth=4)
ax.set_xlabel("Time (Periods)", fontsize=18)
ax.set_xticks(xvals, xlabels, fontsize=16)
ax.tick_params(axis="y", labelsize=16)
ax.set_ylabel("TEMoN-3", fontsize=18)


plt.show()
