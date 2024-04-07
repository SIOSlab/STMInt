import numpy as np
import scipy
from STMint.STMint import STMint
from STMint import TensorNormUtilities as tnu
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# Gateway Nrho ics: transfer sampling error
mu = 1.0 / (81.30059 + 1.0)
x0 = 1.02202151273581740824714855590570360
z0 = 0.182096761524240501132977765539282777
yd0 = -0.103256341062793815791764364248006121
period = 1.5111111111111111111111111111111111111111
transfer_time = period / 10.0

x_0 = np.array([x0, 0, z0, 0, yd0, 0])

# length scale for nondimensional units in km
L = 3.85e5

# Nrho Propagator
integrator = STMint(preset="threeBody", preset_mult=mu, variational_order=2)

x_f, stm, stt = integrator.dynVar_int2(
    [0, transfer_time], x_0, max_step=(transfer_time) / 100.0, output="final"
)

_, stms, stts, ts = integrator.dynVar_int2(
    [0, period], x_0, max_step=(transfer_time) / 100.0, output="all"
)


r_f = x_f[:3]


def calc_error(integrator, stm, transfer_time, r_f, x_0, perturbation):
    delta_r_0 = np.array([*perturbation, 0, 0, 0])

    delta_v_0_1 = -1 * np.array(
        [
            0,
            0,
            0,
            *np.matmul(
                np.linalg.inv(stm[0:3, 3:6]), np.matmul(stm[0:3, 0:3], perturbation)
            ),
        ]
    )

    y_0 = x_0 + delta_r_0 + delta_v_0_1

    r_f_lin = integrator.dynVar_int2([0, transfer_time], y_0, output="final")[0][:3]

    return np.linalg.norm(r_f_lin - r_f, ord=2)


def normalize_sphere_samples(r, n):
    samples = np.random.multivariate_normal([0, 0, 0], np.identity(3), n)

    normalized_samples = []

    for sample in samples:
        normalized_samples.append((sample / np.linalg.norm(sample, ord=2)) * r)

    return normalized_samples


def calc_sphere_max_error(integrator, stm, transfer_time, r_f, x_0, normalized_samples):
    errors = []

    for sample in normalized_samples:
        errors.append(calc_error(integrator, stm, transfer_time, r_f, x_0, sample))

    return np.max(errors)


def calc_f_tensor(stm, stt):
    stm_rr = stm[0:3, 0:3]
    stm_rv = stm[0:3, 3:6]
    inv_stm_rv = np.linalg.inv(stm_rv)

    stt_rrr = stt[0:3, 0:3, 0:3]
    stt_rrv = stt[0:3, 0:3, 3:6]
    stt_rvv = stt[0:3, 3:6, 3:6]
    stt_rvr = stt[0:3, 3:6, 0:3]

    stm_mult = -1.0 * np.matmul(inv_stm_rv, stm_rr)

    first = np.einsum("ijl,lk->ijk", stt_rrv, stm_mult)
    second = np.einsum("ilk,lj->ijk", stt_rvr, stm_mult)
    third = np.einsum("ilp,lj,pk->ijk", stt_rvv, stm_mult, stm_mult)

    F = 0.5 * (stt_rrr + first + second + third)
    FTensSquared = np.einsum("ijk,ilm->jklm", F, F)
    FTensSquared = tnu.symmetrize_tensor(FTensSquared)
    FNormMax = 0
    FArgMaxMax = 0

    for i in range(100):
        FGuess = np.random.multivariate_normal([0, 0, 0], np.identity(3), 1)[0]
        FGuess = FGuess / np.linalg.norm(FGuess, ord=2)
        FArgMax, FNorm = tnu.power_iteration_symmetrizing(
            FTensSquared, FGuess, 100, 1e-9
        )
        if FNorm > FNormMax:
            FNormMax = FNorm
            FArgMaxMax = FArgMax

    return F, FArgMaxMax, FNormMax


s_0yvals = []
s_1yvals = []
s_2yvals = []
s_3yvals = []
m_1yvals = []
m_2yvals = []
m_3yvals = []
xvals = []

F1, F1ArgMax, F1Norm = calc_f_tensor(stm, stt)

# Tensor Norm Calculations

tensor_norms = []

for i in range(1, len(ts)):
    tensor_norms.append(calc_f_tensor(stms[i], stts[i])[2])


for i in range(0, 20):
    # Scale of 500km
    r = 100 * (i + 1) / L
    xvals.append(r)

    # Sampling Method with different number of samples.
    s_0yvals.append(
        calc_sphere_max_error(
            integrator, stm, transfer_time, r_f, x_0, normalize_sphere_samples(r, 200)
        )
    )

    """ Additional max errors for more samples
    s_1yvals.append(
        calc_sphere_max_error(
            stm, transfer_time, r_f, x_0, normalize_sphere_samples(r, 2000)
        )
    )
    s_2yvals.append(
        calc_sphere_max_error(
            stm, transfer_time, r_f, x_0, normalize_sphere_samples(r, 3000)
        )
    )
    s_3yvals.append(
        calc_sphere_max_error(
            stm, transfer_time, r_f, x_0, normalize_sphere_samples(r, 4000)
        )
    )
    """
    # Method 1: Analytical method for calculating maximum error
    m_1yvals.append(pow(r, 2) * np.sqrt(F1Norm))

    # Method 2: Making an educated guess at the maximum error.
    m_2yvals.append(calc_error(integrator, stm, transfer_time, r_f, x_0, r * F1ArgMax))

    # Method 3: Least Squares Error Maximization
    initial_guess = np.array([*(F1ArgMax * r)])

    err = lambda pert: calc_error(integrator, stm, transfer_time, r_f, x_0, pert)
    objective = lambda dr_0: -1.0 * err(dr_0)
    eq_cons = {
        "type": "eq",
        "fun": lambda dr_0: r**2 - np.linalg.norm(dr_0, ord=2) ** 2,
    }

    min = scipy.optimize.minimize(
        objective,
        initial_guess,
        method="SLSQP",
        constraints=eq_cons,
        tol=(1e-18),
        options={"disp": True},
    )

    m_3yvals.append(err(min.x))

# Change normalized units to meters and seconds

xvals = [(x * L) for x in xvals]
# to km
s_0yvals = [(x * L) for x in s_0yvals]
m_1yvals = [(x * L) for x in m_1yvals]
m_2yvals = [(x * L) for x in m_2yvals]
m_3yvals = [(x * L) for x in m_3yvals]
# to m
tensor_norms = [(x / (L)) for x in tensor_norms]


# Changing ts to periods
ts = [(x / period) for x in ts]

# Plotting each method in single graph
plt.style.use("seaborn-v0_8-darkgrid")

fig, axs = plt.subplots(4, sharex=True)
axs[1].plot(xvals, s_0yvals)
axs[1].set_title("Method 0")
axs[0].plot(xvals, m_1yvals)
axs[0].set_title("Method 1")
axs[2].plot(xvals, m_2yvals)
axs[2].set_title("Method 2")
axs[3].plot(xvals, m_3yvals)
axs[3].set_title("Method 3")
axs[3].set_xlabel("Radius of Relative Initial Position (km)", fontsize=16)
fig.text(
    0.06,
    0.5,
    "Maximum Error (km)",
    ha="center",
    va="center",
    rotation="vertical",
    fontsize=16,
)
plt.subplots_adjust(hspace=1, left=0.2, right=0.9)

# Plotting only method 3
fig2, model3 = plt.subplots(figsize=(8, 6))
model3.plot(xvals, m_3yvals)
model3.set_xlabel("Radius of Relative Initial Position (km)", fontsize=18)
model3.set_ylabel("Maximum Error (km)", fontsize=18)
model3.tick_params(labelsize=14)


# Plotting error between methods (0, 1, and 2 with respect to 3)
error0_3 = []
error1_3 = []
error2_3 = []
for i in range(len(xvals)):
    error0_3.append((abs((s_0yvals[i] - m_3yvals[i])) / m_3yvals[i]) * 100)
    error1_3.append((abs((m_1yvals[i] - m_3yvals[i])) / m_3yvals[i]) * 100)
    error2_3.append((abs((m_2yvals[i] - m_3yvals[i])) / m_3yvals[i]) * 100)

fig3, error = plt.subplots(figsize=(8, 6))
error.plot(xvals, error0_3, label="Sampling")
error.plot(xvals, error1_3, label="Tensor Norm")
#below 10^-5 level
#error.plot(xvals, error2_3, label="Eigenvec. Eval.")
error.set_xlabel("Radius of Relative Initial Position (km)", fontsize=18)
error.set_ylabel("Method Percentage Error", fontsize=18)
error.set_yscale("log")
error.legend(fontsize=14)
error.tick_params(labelsize=14)

fig4, norms = plt.subplots(figsize=(8, 6))
norms.plot(ts[21:], tensor_norms[20:])
norms.set_xlabel("Time of Flight (periods)", fontsize=18)
norms.set_ylabel("Tensor Norm (km^-1)", fontsize=18)
norms.set_yscale("log")
norms.tick_params(labelsize=14)

plt.show()

plt.show()
