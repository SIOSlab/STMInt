import numpy as np
import scipy
from astropy import units as u
from STMint.STMint import STMint
from STMint import TensorNormUtilities as tnu
from poliastro.twobody.orbit import Orbit
import poliastro.bodies as body
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# ISS Keplerian Elements
a = 6738 << u.km
ecc = 0.0005140 << u.one
inc = 51.6434 << u.deg

# Since these elements are constantly changing, they will be ignored for simplicity
raan = 0 << u.deg
argp = 0 << u.deg
nu = 0 << u.deg

iss_orbit = Orbit.from_classical(body.Earth, a, ecc, inc, raan, argp, nu)
# ISS ICS
x_0 = np.array([*iss_orbit.r.value, *iss_orbit.v.value])


def calc_error(stm, transfer_time, x_0, perturbation):
    iss_reference_orbit = iss_orbit.propagate(transfer_time * u.s)

    r_f_ref = np.array([*iss_reference_orbit.r.value])

    delta_x_0 = np.array([0, 0, 0, *perturbation])

    y_0 = x_0 + delta_x_0

    iss_perturbed_orbit = Orbit.from_vectors(
        body.Earth, y_0[:3] * u.km, y_0[3:] * u.km / u.s
    )

    r_f_pert = np.array([iss_perturbed_orbit.propagate(transfer_time * u.s).r.value])

    return np.linalg.norm(((r_f_pert - r_f_ref) - np.matmul(stm, delta_x_0)[:3]), ord=2)


def normalize_sphere_samples(r, n):
    samples = np.random.multivariate_normal([0, 0, 0], np.identity(3), n)

    normalized_samples = []

    for sample in samples:
        normalized_samples.append((sample / np.linalg.norm(sample, ord=2)) * r)

    return normalized_samples


def calc_sphere_max_error(stm, transfer_time, x_0, normalized_samples):
    errors = []

    for sample in normalized_samples:
        errors.append(calc_error(stm, transfer_time, x_0, sample))

    return np.max(errors)


s_0yvals = []
s_1yvals = []
s_2yvals = []
s_3yvals = []
m_1yvals = []
m_2yvals = []
m_3yvals = []
xvals = []
transfer_time = iss_orbit.period.to(u.s).value / 10.0

integrator = STMint(preset="twoBodyEarth", variational_order=2)
stm = integrator.dynVar_int([0, transfer_time], x_0, output="final")[1]
stt = integrator.dynVar_int2([0, transfer_time], x_0, output="final")[2]

for i in range(0, 25):
    # Convert to internal units of km (range of ~100m/s)
    r = (10 * (i + 1)) / 1000
    xvals.append(r)

    sttArgMax, m_1norm = tnu.stt_2_norm(stm[:3, 3:], stt[:3, 3:, 3:])

    # Sampling Method with different number of samples.
    s_0yvals.append(
        calc_sphere_max_error(
            stm, transfer_time, x_0, normalize_sphere_samples(r, 5000)
        )
    )

    """"
    s_1yvals.append(
        calc_sphere_max_error(
            stm, transfer_time, x_0, normalize_sphere_samples(r, 2000)
        )
    )
    s_2yvals.append(
        calc_sphere_max_error(
            stm, transfer_time, x_0, normalize_sphere_samples(r, 3000)
        )
    )
    s_3yvals.append(
        calc_sphere_max_error(
            stm, transfer_time, x_0, normalize_sphere_samples(r, 4000)
        )
    )
    """

    # Method 1: Analytical method for calculating maximum error
    m_1yvals.append(0.5 * pow(r, 2) * m_1norm)

    # Method 2: Making an educated guess at the maximum error.
    m_2yvals.append(calc_error(stm, transfer_time, x_0, r * sttArgMax))

    # Method 3: Least Squares Error Maximization
    initial_guess = np.array([*(sttArgMax * r)])

    err = lambda pert: calc_error(stm, transfer_time, x_0, pert)
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
        options={"ftol": 1e-12, "disp": True},
    )

    m_3yvals.append(err(min.x))

# Change units from km to meters
xvals = [x * 1000 for x in xvals]
s_0yvals = [x * 1000 for x in s_0yvals]
m_1yvals = [x * 1000 for x in m_1yvals]
m_2yvals = [x * 1000 for x in m_2yvals]
m_3yvals = [x * 1000 for x in m_3yvals]

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
model3.plot(xvals, m_1yvals)
model3.set_xlabel("Radius of Sphere of Perturbation (m/s)", fontsize=16)
model3.set_ylabel("Maximum Error (m)", fontsize=16)

# Plotting error between methods (0, 1, and 2 with respect to 3)
error0_3 = []
error1_3 = []
error2_3 = []

for i in range(len(xvals)):
    error0_3.append((abs((s_0yvals[i] - m_3yvals[i])) / m_3yvals[i]) * 100)
    error1_3.append((abs((m_1yvals[i] - m_3yvals[i])) / m_3yvals[i]) * 100)
    error2_3.append((abs((m_2yvals[i] - m_3yvals[i])) / m_3yvals[i]) * 100)

fig3, error = plt.subplots(figsize=(7, 4.8))
error.plot(xvals, error0_3, label="Methods 0 and 3")
error.plot(xvals, error1_3, label="Methods 1 and 3")
error.plot(xvals, error2_3, label="Methods 2 and 3")
error.set_xlabel("Radius of Sphere of Perturbation (m/s)", fontsize=16)
error.set_ylabel("Method Percentage Error", fontsize=16)
error.set_yscale("log")
error.legend()

plt.show()
