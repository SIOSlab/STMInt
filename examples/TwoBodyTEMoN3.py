import numpy as np
from STMint.STMint import STMint
from STMint import TensorNormUtilities as tnu
import math
import matplotlib.pyplot as plt
import numpy.linalg


def normalize_sphere_samples(r, n):
    samples = np.random.multivariate_normal([0, 0, 0, 0, 0, 0], np.identity(6), n)

    normalized_samples = []

    for sample in samples:
        normalized_samples.append((sample / np.linalg.norm(sample, ord=2)) * r)

    return normalized_samples


def calc_TEMoN3_sampling(stm, stt, num_samples):
    temons = []
    normalized_samples = normalize_sphere_samples(1., num_samples)
    #CGT3 = 0.5 * (np.einsum("lij,lk->ijk", stt, stm) + np.einsum("ljk,li->ijk", stt, stm))
    CGT3 = np.einsum("lij,lk->ijk", stt, stm)

    for sample in normalized_samples:
        temon1 = np.sqrt(np.abs(np.einsum("ijk,i,j,k->", CGT3, sample, sample, sample)) / (np.linalg.norm(np.einsum("ij,j->i", stm, sample))**2))
        temon = np.sqrt(np.abs(np.dot(np.einsum("ijk,j,k->i",stt, sample, sample), np.matmul(stm, sample)))) / np.linalg.norm(np.einsum("ij,j->i", stm, sample))
        temons.append(temon)

    return np.max(temons)

# Calculating the solutions to non-dimensional two-body motion
integ = STMint(preset="twoBody", variational_order=2)

# integrate variational equations
# 10 periods of a circular orbit with unit radius
states, stms, stts, ts = integ.dynVar_int2(
    [0, (4.0 * math.pi)], [1, 0, 0, 0, 1, 0], output="all", max_step=0.1
)

NLIs = []
NLIsSampling = []

# Eliminating first element to not divide by zero in following functions
ts = ts[1:]
for i in range(len(ts)):
    i = i + 1
    NLIs.append(np.sqrt(tnu.nonlin_index_TEMoN3(stms[i], stts[i])))
    #NLIsSampling.append(calc_TEMoN3_sampling(stms[i], stts[i], 1000))
    print(NLIs[-1])
    print(NLIsSampling[-1])

# Plotting
xvals = []
for i in range(0, 5, 2):
    xvals.append(math.pi * i)

xlabels = [0]
for i in range(2, 5, 2):
    xlabels.append(str(i) + r"$\pi$")


plt.figure(figsize=(8, 6))
plt.plot(ts, NLIs)
#plt.plot(ts, NLIsSampling)
plt.title("Nonlinearity Associated With Non-dimensional Circular Two-Body Motion")
plt.xlabel("Time")
plt.xticks(xvals, xlabels)
plt.ylabel("Nonlinearity Index")


plt.show()