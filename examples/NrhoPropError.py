import numpy as np
import scipy
from astropy import units as u
from STMint.STMint import STMint
from STMint import TensorNormUtilities as tnu
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# ======================================================================================================================
# Method 0: Sampling method for calculating maximum error
# ======================================================================================================================

# Gateway Nrho ics: transfer sampling error
mu = 1.0 / (81.30059 + 1.0)
x0 = 1.02202151273581740824714855590570360
z0 = 0.182096761524240501132977765539282777
yd0 = -0.103256341062793815791764364248006121
period = 1.5111111111111111111111111111111111111111

x_0 = np.array([x0, 0, z0, 0, yd0, 0])

# Nrho Propagator
integrator = STMint(preset="threeBody", preset_mult=mu, variational_order=2)


def calc_error(stm, transfer_time, r_f, x_0, perturbation):
    delta_x_0 = np.array([0, 0, 0, *perturbation])

    y_0 = x_0 + delta_x_0

    r_f_pert = integrator.dynVar_int2([0, transfer_time], y_0, output="final")[0][:3]

    return np.linalg.norm(((r_f_pert - r_f) - np.matmul(stm, delta_x_0)[:3]), ord=2)


# Step 3


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
transfer_time = period / 10.0

x_f, stm, stt = integrator.dynVar_int2([0, transfer_time], x_0, output="final")
r_f = x_f[:3]

for i in range(0, 10):
    # Change so r is linearly distributed
    r = np.linalg.norm(x_0[3:]) / (100000) * ((i + 1) * 50)
    xvals.append(r)
    sttArgMax, m_1norm = tnu.stt_2_norm(stm[:3, 3:], stt[:3, 3:, 3:])

    # Method 0: Sampling
    s_0yvals.append(
        calc_sphere_max_error(
            stm, transfer_time, r_f, x_0, normalize_sphere_samples(r, 100)
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
        options={"ftol": 1e-9, "disp": True},
    )

    m_3yvals.append(err(min.x))

#   Change xvals' units to meters
xvals_m = [x * 1000 for x in xvals]
m_3yvals_m = [x * 1000 for x in m_3yvals]

# Plotting each method in single graph
fig, axs = plt.subplots(4, sharex=True)
axs[1].plot(xvals_m, s_0yvals)
axs[1].set_title("Method 0")
axs[0].plot(xvals_m, m_1yvals)
axs[0].set_title("Method 1")
axs[2].plot(xvals_m, m_2yvals)
axs[2].set_title("Method 2")
axs[3].plot(xvals_m, m_3yvals)
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
fig2, model3 = plt.subplots(figsize=(8, 4.8))
model3.plot(xvals_m, m_3yvals_m)
model3.set_xlabel("Radius of Sphere of Perturbation (m/s)", fontsize=16)
model3.set_ylabel("Maximum Error (m)", fontsize=16)

# Plotting error between methods (1 and 2 with respect to 3)
error1_3 = []
error2_3 = []
for i in range(len(xvals_m)):
    error1_3.append((abs((m_1yvals[i] - m_3yvals[i])) / m_3yvals[i]) * 100)
    error2_3.append((abs((m_2yvals[i] - m_3yvals[i])) / m_3yvals[i]) * 100)

fig3, error = plt.subplots(figsize=(7, 4.8))
error.plot(xvals_m, error1_3, label="Methods 1 and 3")
error.plot(xvals_m, error2_3, label="Methods 2 and 3")
error.set_xlabel("Radius of Sphere of Perturbation (m/s)", fontsize=16)
error.set_ylabel("Method Percentage Error", fontsize=16)
error.legend()

plt.show()
