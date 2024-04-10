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

# Dimensionalized Unit Conversion (Koon Lo Marsden pg. 25, Earth-Moon)

L = 3.85e5
V = 1.025
T = 2.361e6 / 2. / np.pi

x_0 = np.array([x0, 0, z0, 0, yd0, 0])

# Nrho Propagator
integrator = STMint(preset="threeBody", preset_mult=mu, variational_order=2)

transfer_time = period * 0.1

x_f, stm, stt = integrator.dynVar_int2([0, transfer_time], x_0, output="final")
r_f = x_f[:3]


def calc_error(stm, transfer_time, r_f, x_0, perturbation):
    delta_r_f_star = perturbation

    delta_v_0_1 = np.matmul(np.linalg.inv(stm[0:3, 3:6]), delta_r_f_star)

    y_0 = x_0 + np.array([0, 0, 0, *delta_v_0_1])

    r_f_1 = integrator.dynVar_int2([0, transfer_time], y_0, output="final")[0][:3]

    delta_r_f_1 = r_f_1 - r_f

    return np.linalg.norm(delta_r_f_star - delta_r_f_1, ord=2)


def normalize_sphere_samples(r, n):
    samples = np.random.multivariate_normal([0, 0, 0], np.identity(3), n)

    normalized_samples = []

    for sample in samples:
        normalized_samples.append((sample / np.linalg.norm(sample, ord=2)) * r)

    return normalized_samples


def calc_sphere_max_error(stm, transfer_time, r_f, x_0, normalized_samples):
    errors = []

    for sample in normalized_samples:
        errors.append(calc_error(stm, transfer_time, r_f, x_0, sample))

    return np.max(errors)


def calc_e_tensor(stm, stt):
    stm_rv = stm[0:3, 3:6]
    stt_rvv = stt[0:3, 3:6, 3:6]
    inv_stm_rv = np.linalg.inv(stm_rv)

    E = 0.5 * np.einsum("ilm,lj,mk->ijk", stt_rvv, inv_stm_rv, inv_stm_rv)
    tensSquared = np.einsum("ijk,ilm->jklm", E, E)
    ENormMax = 0
    EArgMaxMax = 0
    # try 10 different initial guesses for symmetric higher order power iteration
    for i in range(10):
        Eguess = np.random.multivariate_normal([0, 0, 0], np.identity(3), 1)[0]
        Eguess = Eguess / np.linalg.norm(Eguess, ord=2)
        EArgMax, ENorm = tnu.power_iteration_symmetrizing(
            tensSquared, Eguess, 100, 1e-9
        )
        if ENorm > ENormMax:
            ENormMax = ENorm
            EArgMaxMax = EArgMax
    return E, EArgMaxMax, ENormMax


s_0yvals = []
s_1yvals = []
s_2yvals = []
s_3yvals = []
m_1yvals = []
m_2yvals = []
m_3yvals = []
xvals = []

E1, E1ArgMax, E1Norm = calc_e_tensor(stm, stt)

# Tensor Norm Calculations

tensor_norms = []

_, stms, stts, ts = integrator.dynVar_int2(
    [0, period], x_0, max_step=(transfer_time) / 100.0, output="all"
)

for i in range(1, len(ts)):
    tensor_norms.append(calc_e_tensor(stms[i], stts[i])[2])

for i in range(0, 20):
    # Scale of 2000km
    r = 100.0 * (i + 1) / L
    xvals.append(r)

    # Method 0: Sampling
    s_0yvals.append(
        calc_sphere_max_error(
            stm, transfer_time, r_f, x_0, normalize_sphere_samples(r, 5000)
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
    m_1yvals.append(pow(r, 2) * np.sqrt(E1Norm))

    # Method 2: Making an educated guess at the maximum error.
    err_eval1 = calc_error(stm, transfer_time, r_f, x_0, r * E1ArgMax)
    err_eval2 = calc_error(stm, transfer_time, r_f, x_0, -1. * r * E1ArgMax)
    m_2yvals.append(max(err_eval1, err_eval2))

    # Method 3: Least Squares Error Maximization
    if err_eval1 > err_eval2:
        initial_guess = np.array([*(E1ArgMax * r)])
    else:
        initial_guess = np.array([*(-1. * E1ArgMax * r)])

    err = lambda pert: calc_error(stm, transfer_time, r_f, x_0, pert)
    objective = lambda dr_f: -1.0 * err(dr_f)
    eq_cons = {
        "type": "eq",
        "fun": lambda dr_f: r**2 - np.linalg.norm(dr_f, ord=2) ** 2,
    }

    min = scipy.optimize.minimize(
        objective,
        initial_guess,
        method="SLSQP",
        constraints=eq_cons,
        tol=1e-12,
        options={"disp": True},
    )

    m_3yvals.append(err(min.x))

# Change normalized units to meters and seconds

xvals = [(x * L) for x in xvals]
s_0yvals = [(x * L) for x in s_0yvals]
m_1yvals = [(x * L) for x in m_1yvals]
m_2yvals = [(x * L) for x in m_2yvals]
m_3yvals = [(x * L) for x in m_3yvals]
tensor_norms = [(x / (L * 1000)) for x in tensor_norms]

# Changing normalized units to periods
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
axs[3].set_xlabel("Radius of Relative Final Position (km)", fontsize=16)
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
model3.plot(xvals, np.array(m_3yvals) * 1000)
model3.set_xlabel("Radius of Relative Final Position (km)", fontsize=18)
model3.set_ylabel("Maximum Error (m)", fontsize=18)
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
#below 10^-7 level
#error.plot(xvals, error2_3, label="Eigenvec. Eval")
error.set_xlabel("Radius of Relative Final Position (km)", fontsize=18)
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
