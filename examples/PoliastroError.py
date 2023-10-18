import numpy as np
import scipy
from astropy import units as u
from STMint.STMint import STMint
from STMint import TensorNormUtilities as tnu
from poliastro.twobody.orbit import Orbit
import poliastro.bodies as body
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import least_squares

# ======================================================================================================================
# Method 0: Sampling method for calculating maximum error
# ======================================================================================================================

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


# Step 3


def normalize_sphere_samples(r, n):
    samples = np.random.multivariate_normal([0, 0, 0], np.identity(3), n)

    normalized_samples = []

    for sample in samples:
        normalized_samples.append((sample / np.linalg.norm(samples, ord=2)) * r)

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

for i in range(0, 2):
    # Change so r is linearly distributed
    r = np.linalg.norm(x_0[3:]) / (100000) * ((i + 1) * 50)
    xvals.append(r)
    sttArgMax, m_1norm = tnu.stt_2_norm(stm[:3, 3:], stt[:3, 3:, 3:])

    # Sampling Method with different number of samples.
    s_0yvals.append(
        calc_sphere_max_error(stm, transfer_time, x_0, normalize_sphere_samples(r, 50))
    )
    s_1yvals.append(
        calc_sphere_max_error(stm, transfer_time, x_0, normalize_sphere_samples(r, 250))
    )
    s_2yvals.append(
        calc_sphere_max_error(stm, transfer_time, x_0, normalize_sphere_samples(r, 750))
    )
    s_3yvals.append(
        calc_sphere_max_error(
            stm, transfer_time, x_0, normalize_sphere_samples(r, 2250)
        )
    )

    # Method 1: Analytical method for calculating maximum error
    m_1yvals.append(0.5 * pow(r, 2) * m_1norm)

    m_2yvals.append(calc_error(stm, transfer_time, x_0, r * sttArgMax))

    # Method 3: Least Squares Error Maximization
    initial_guess = np.array([*(sttArgMax * r)])

    err = lambda pert: calc_error(stm, transfer_time, x_0, pert)
    objective = lambda dv0: err(r * sttArgMax) - err(dv0)
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


# Plotting each method in single graph
fig, axs = plt.subplots(4, sharex=True)
axs[1].plot(xvals, s_0yvals)
axs[1].set_title("Method 0")
axs[0].plot(xvals, m_1yvals)
axs[0].set_title("Method 1")
axs[2].plot(xvals, m_2yvals)
axs[2].set_title("Method 2")
axs[3].plot(xvals, m_3yvals)
axs[3].set_title("Method 3")
axs[0].set_xlabel("Radius of Sphere of Perturbation", fontsize=16)
fig.text(
    0.06,
    0.5,
    "Maximum Error",
    ha="center",
    va="center",
    rotation="vertical",
    fontsize=16,
)
plt.subplots_adjust(hspace=1, left=0.2, right=0.9)

# Plotting only method 3
fig2, model3 = plt.subplots(figsize=(8, 4.8))
model3.plot(xvals, m_3yvals)
model3.set_xlabel("Radius of Sphere of Perturbation (km)", fontsize=16)
model3.set_ylabel("Maximum Error", fontsize=16)

# Plotting error between methods (1 and 2 with resepct to 3)
error1_3 = []
error2_3 = []
for i in range(len(xvals)):
    error1_3.append((abs((m_1yvals[i] - m_3yvals[i])) / m_3yvals[i]) * 100)
    error2_3.append((abs((m_2yvals[i] - m_3yvals[i])) / m_3yvals[i]) * 100)

fig3, error = plt.subplots(figsize=(7, 4.8))
error.plot(xvals, error1_3, label="Methods 1 and 3")
error.plot(xvals, error2_3, label="Methods 2 and 3")
error.set_xlabel("Radius of Sphere of Perturbation (km)", fontsize=16)
error.set_ylabel("Method Percentage Error", fontsize=16)
error.legend()

# Plotting error between sampling method and method 3
s_error0_3 = []
s_error1_3 = []
s_error2_3 = []
s_error3_3 = []
for i in range(len(s_0yvals)):
    s_error0_3.append((abs((s_0yvals[i] - m_3yvals[i])) / m_3yvals[i]) * 100)
    s_error1_3.append((abs((s_1yvals[i] - m_3yvals[i])) / m_3yvals[i]) * 100)
    s_error2_3.append((abs((s_2yvals[i] - m_3yvals[i])) / m_3yvals[i]) * 100)
    s_error3_3.append((abs((s_3yvals[i] - m_3yvals[i])) / m_3yvals[i]) * 100)


fig4, s_error = plt.subplots(figsize=(7, 4.8))
s_error.plot(xvals, s_error0_3, label="50 Samples")
s_error.plot(xvals, s_error1_3, label="500 Samples")
s_error.plot(xvals, s_error2_3, label="1000 Samples")
s_error.plot(xvals, s_error3_3, label="2500 Samples")
s_error.set_xlabel("Radius of Sphere of Perturbation (km)", fontsize=16)
s_error.set_ylabel("Method Percentage Error", fontsize=16)
s_error.legend()

plt.show()
