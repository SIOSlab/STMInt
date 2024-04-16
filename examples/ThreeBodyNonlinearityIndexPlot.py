import numpy as np
from STMint.STMint import STMint
from STMint import TensorNormUtilities as tnu
import math
import matplotlib.pyplot as plt
import numpy.linalg

# Gateway Nrho ics: transfer sampling error
mu = 1.0 / (81.30059 + 1.0)
x0 = 1.02202151273581740824714855590570360
z0 = 0.182096761524240501132977765539282777
yd0 = -0.103256341062793815791764364248006121
period = 1.5111111111111111111111111111111111111111

x_0 = np.array([x0, 0, z0, 0, yd0, 0])

# Nrho Propagator
integ = STMint(preset="threeBody", preset_mult=mu, variational_order=2)

# integrate variational equations
# 10 periods of a circular orbit with unit radius
states, stms, stts, ts = integ.dynVar_int2(
    [0, period], x_0, output="all", max_step=0.005
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

# Plotting
xvals = np.array([0, 0.25, 0.5, 0.75, 1]) * period
# for i in range(0, 3, 1):
#    xvals.append(i * period)

xlabels = ["0", "1/4", "1/2", "3/4", "1"]
# for i in range(1, 3, 1):
#    xlabels.append(str(i))

plt.style.use("seaborn-v0_8-colorblind")

fig, ax = plt.subplots(figsize=(8, 6))
ax.semilogy(ts, NLI1s, label="2-Norm Bound", linewidth=4)
ax.semilogy(ts, NLI2s, label="(\u221e, 2)-Norm", linewidth=4)
ax.semilogy(ts, NLI3s, label="2-Norm", linewidth=4)
ax.semilogy(ts, NLI4s, label="(Frobenius, 2)-Norm", linewidth=4)
# ax.set_title("Nonlinearity Associated With Non-dimensional Circular Two-Body Motion")
ax.set_xlabel("Time (Periods)", fontsize=18)
ax.set_xticks(xvals, xlabels, fontsize=16)
ax.tick_params(axis="y", labelsize=16)
ax.set_ylabel("Nonlinearity Index", fontsize=18)
ax.legend(fontsize=12)

fig, ax = plt.subplots(figsize=(8, 6))
ax.semilogy(ts, NLI1s, label="2-Norm Bound", linewidth=4)
ax.semilogy(ts, NLI2s, label="(\u221e, 2)-Norm", linewidth=4)
ax.semilogy(ts, NLI3s, label="2-Norm", linewidth=4)
ax.semilogy(ts, NLI4s, label="(Frobenius, 2)-Norm", linewidth=4)
ax.semilogy(ts, NLI6s, label="TEMoN-3", linewidth=4)
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
ax.semilogy(ts, NLI5s, linewidth=4)
ax.set_xlabel("Time (Periods)", fontsize=18)
ax.set_xticks(xvals, xlabels, fontsize=16)
ax.tick_params(axis="y", labelsize=16)
ax.set_ylabel("DEMoN-2", fontsize=18)

fig, ax = plt.subplots(figsize=(8, 6))
ax.semilogy(ts, NLI6s, linewidth=4)
ax.set_xlabel("Time (Periods)", fontsize=18)
ax.set_xticks(xvals, xlabels, fontsize=16)
ax.tick_params(axis="y", labelsize=16)
ax.set_ylabel("DEMoN-2", fontsize=18)


plt.show()
