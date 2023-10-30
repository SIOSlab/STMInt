import numpy as np
from astropy import units as u
from STMint.STMint import STMint
from STMint import TensorNormUtilities as tnu
from poliastro.twobody.orbit import Orbit
import poliastro.bodies as body
import matplotlib.pyplot as plt


def calc_error(stm, transfer_time, r_f, x_0, perturbation):
    delta_r_f_star = perturbation

    delta_v_0_1 = np.matmul(np.linalg.inv(stm[0:3, 3:6]), delta_r_f_star)

    y_0 = x_0 + np.array([0, 0, 0, *delta_v_0_1])

    iss_approx_orbit = Orbit.from_vectors(
        body.Earth, y_0[:3] * u.km, y_0[3:] * u.km / u.s
    )

    r_f_1 = np.array([iss_approx_orbit.propagate(transfer_time * u.s).r.value])

    delta_r_f_1 = np.abs(r_f - r_f_1)

    return np.abs(delta_r_f_star - delta_r_f_1)


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
    return np.einsum(
        "ilm,lj,mk->ijk", stt_rvv, np.linalg.inv(stm_rv), np.linalg.inv(stm_rv)
    )


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

transfer_time = iss_orbit.period.to(u.s).value / 10.0

iss_reference_orbit = iss_orbit.propagate(transfer_time * u.s)

r_f = np.array([*iss_reference_orbit.r.value])

s_0yvals = []
s_1yvals = []
s_2yvals = []
s_3yvals = []
m_1yvals = []
xvals = []

integrator = STMint(preset="twoBodyEarth", variational_order=2)
stm = integrator.dynVar_int([0, transfer_time], x_0, output="final")[1]
stt = integrator.dynVar_int2([0, transfer_time], x_0, output="final")[2]

E1 = calc_e_tensor(stm, stt)

E1guess = np.array([1, 1, 1]) / np.linalg.norm(np.array([1, 1, 1]), ord=2)
tensSquared = np.einsum("ijk,ilm->jklm", E1, E1)
E1ArgMax, E1Norm = tnu.power_iteration_symmetrizing(tensSquared, E1guess, 20, 1e-3)

for i in range(0, 20):
    # Change so r is linearly distributed
    r = 0.5 * (i + 1)
    xvals.append(r)

    # Sampling Method with different number of samples.
    s_0yvals.append(
        calc_sphere_max_error(
            stm, transfer_time, r_f, x_0, normalize_sphere_samples(r, 1000)
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
    m_1yvals.append(pow(r, 2) * E1Norm)

# Change xvals' units to meters
# xvals_m = [x * 1000 for x in xvals]

# Plotting each method in single graph
fig, axs = plt.subplots(2, sharex=True)
axs[0].plot(xvals, s_0yvals)
axs[0].set_title("Method 0")
axs[1].plot(xvals, m_1yvals)
axs[1].set_title("Method 1")
axs[1].set_xlabel("Radius of Relative Final Position (km)", fontsize=16)
fig.text(
    0.06,
    0.5,
    "Maximum Error (km)",
    ha="center",
    va="center",
    rotation="vertical",
    fontsize=16,
)

# Plotting only method 1
fig2, model3 = plt.subplots(figsize=(8, 4.8))
model3.plot(xvals, m_1yvals)
model3.set_xlabel("Radius of Relative Final Position (km)", fontsize=16)
model3.set_ylabel("Maximum Error (km)", fontsize=16)

plt.show()
