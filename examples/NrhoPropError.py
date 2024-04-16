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

# Nrho Propagator
integrator = STMint(preset="threeBody", preset_mult=mu, variational_order=2)

x_f, stm, stt = integrator.dynVar_int2(
    [0, transfer_time], x_0, max_step=(transfer_time) / 100.0, output="final"
)

_, stms, stts, ts = integrator.dynVar_int2(
    [0, period], x_0, max_step=(transfer_time) / 100.0, output="all"
)


r_f = x_f[:3]

tensor_norms = []
for i in range(1, len(ts)):
    tensor_norms.append(tnu.stt_2_norm(stms[i][:3, 3:], stts[i][:3, 3:, 3:])[1])


# Dimensionalized Unit Conversion (Koon Lo Marsden pg. 25, Earth-Moon)

L = 3.85e5  # Distance (km)
V = 1.025  # Velocity (km / s)
T = 2.361e6 / 2.0 / np.pi  # Time (Orbital period)


def calc_error(stm, transfer_time, r_f, x_0, perturbation):
    delta_x_0 = np.array([0, 0, 0, *perturbation])

    y_0 = x_0 + delta_x_0

    # r_f_pert = integrator.dynVar_int2([0, transfer_time], y_0, output="final")[0][:3]
    r_f_pert = integrator.dyn_int([0, transfer_time], y_0).y[:3, -1]

    return np.linalg.norm(((r_f_pert - r_f) - np.matmul(stm, delta_x_0)[:3]), ord=2)


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


s_0yvals = []
m_1yvals = []
m_2yvals = []
m_3yvals = []
xvals = []


for i in range(0, 20):
    # Multiply by V to achieve normalized system
    r = (10 * (i + 1)) / (V * 1000)
    xvals.append(r)
    sttArgMax, m_1norm = tnu.stt_2_norm(stm[:3, 3:], stt[:3, 3:, 3:])

    # Method 0: Sampling
    s_0yvals.append(
        calc_sphere_max_error(
            stm, transfer_time, r_f, x_0, normalize_sphere_samples(r, 5000)
        )
    )

    # Method 1: Analytical method for calculating maximum error
    m_1yvals.append(0.5 * pow(r, 2) * m_1norm)

    # Method 2: Making an educated guess at the maximum error.
    m_2yvals.append(calc_error(stm, transfer_time, r_f, x_0, r * sttArgMax))

    # Method 3: Least Squares Error Maximization
    initial_guess = np.array([*(sttArgMax * r)])

    err = lambda pert: calc_error(stm, transfer_time, r_f, x_0, pert)
    objective = lambda dv0: -1.0 * err(dv0)
    eq_cons = {
        "type": "eq",
        "fun": lambda dv0: r**2 - np.linalg.norm(dv0, ord=2) ** 2,
    }

    min = scipy.optimize.minimize(
        objective,
        initial_guess,
        method="SLSQP",
        constraints=eq_cons,
        tol=1e-17,
        options={"disp": True},
    )

    m_3yvals.append(err(min.x))

# Change normalized units to meters and seconds

xvals = [(x * V) * 1000 for x in xvals]
s_0yvals = [(x * L) * 1000 for x in s_0yvals]
m_1yvals = [(x * L) * 1000 for x in m_1yvals]
m_2yvals = [(x * L) * 1000 for x in m_2yvals]
m_3yvals = [(x * L) * 1000 for x in m_3yvals]
tensor_norms = [((x * (T**2)) / (L * 1000)) for x in tensor_norms]

# Change integrator time-step to periods

ts = [(x / period) for x in ts]

# Plotting each method in single graph
plt.style.use("seaborn-v0_8-colorblind")

fig, axs = plt.subplots(4, sharex=True)
axs[1].plot(xvals, s_0yvals)
axs[1].set_title("Method 0")
axs[0].plot(xvals, m_1yvals)
axs[0].set_title("Method 1")
axs[2].plot(xvals, m_2yvals)
axs[2].set_title("Method 2")
axs[3].plot(xvals, m_3yvals)
axs[3].set_title("Method 3")
axs[3].set_xlabel("Radius of Sphere of Perturbation (m/s)", fontsize=16)
fig.text(
    0.06,
    0.5,
    "Maximum Error (m)",
    ha="center",
    va="center",
    rotation="vertical",
    fontsize=16,
)
plt.subplots_adjust(hspace=1, left=0.2, right=0.9)

# Plotting only method 3
fig2, model3 = plt.subplots(figsize=(8, 6))
model3.plot(xvals, np.divide(m_1yvals, 1000.0), linewidth=4)
model3.set_xlabel("Radius of Sphere of Perturbation (m/s)", fontsize=18)
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
error.plot(xvals, error0_3, label="Sampling", linewidth=4)
error.plot(xvals, error1_3, label="Tensor Norm", linewidth=4)
error.plot(xvals, error2_3, label="Eigenvec. Eval.", linewidth=4)
error.set_xlabel("Radius of Sphere of Perturbation (m/s)", fontsize=18)
error.set_ylabel("Method Percentage Error", fontsize=18)
error.set_yscale("log")
error.legend(fontsize=14)
error.tick_params(labelsize=14)

xvals = np.array([0, 0.25, 0.5, 0.75, 1])
xlabels = ["0", "1/4", "1/2", "3/4", "1"]

fig4, norms = plt.subplots(figsize=(8, 6))
norms.plot(ts[21:], tensor_norms[20:], linewidth=4)
norms.set_xlabel("Time of Flight (periods)", fontsize=18)
norms.set_ylabel("Tensor Norm (s^2 / m)", fontsize=18)
norms.set_yscale("log")
norms.tick_params(labelsize=14)
norms.set_xticks(xvals, xlabels, fontsize=14)

fig2.savefig("figures/Prop/threeBodyPropOpt.png")
fig3.savefig("figures/Prop/threeBodyPropError.png")
fig4.savefig("figures/Prop/threeBodyPropTNorms.png")
