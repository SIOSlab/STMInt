import numpy as np
from astropy import units as u
from STMint.STMint import STMint
from STMint import TensorNormUtilities as tnu
from poliastro.twobody.orbit import Orbit
import poliastro.bodies as body
import matplotlib.pyplot as plt

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

    iss_perturbed_orbit = Orbit.from_vectors(body.Earth, y_0[:3] * u.km, y_0[3:] * u.km / u.s)

    r_f_pert = np.array([iss_perturbed_orbit.propagate(transfer_time * u.s).r.value])

    return np.linalg.norm(((r_f_ref - r_f_pert) - np.matmul(stm, delta_x_0)[:3]), ord=2)

# Step 3

def calc_sphere_max_error(stm, transfer_time, x_0, r, n):
    samples = np.random.multivariate_normal([0, 0, 0], np.identity(3), n)

    normalized_samples = []

    for sample in samples:
        normalized_samples.append((sample / np.linalg.norm(samples, ord=2)) * r)

    errors = []

    for sample in normalized_samples:
        errors.append(calc_error(stm, transfer_time, x_0, sample))

    return np.max(errors)

m_0yvals = []
m_1yvals = []
m_2yvals = []
xvals = []
transfer_time = iss_orbit.period.to(u.s).value/10.0

# ======================================================================================================================
# Method 1: Analytical method for calculating maximum error
# ======================================================================================================================

# Fix this when change dynVar_int2 output
integrator = STMint(preset="twoBodyEarth", variational_order=2)
stm = integrator.dynVar_int([0, transfer_time], x_0, output="final")[1]
stt = integrator.dynVar_int2([0, transfer_time], x_0, output="final")[2]

for i in range(1, 20):
    # Change so r is linearly distributed
    r = np.linalg.norm(x_0[3:]) / (100000) * (i * 50)
    xvals.append(r)
    n = 500
    m_2argmax, m_1norm = tnu.stt_2_norm(stm[:3, 3:], stt[:3, 3:, 3:])

    m_0yvals.append(calc_sphere_max_error(stm, transfer_time, x_0, r, n))
    m_1yvals.append(.5 * pow(r,2) * m_1norm)
    m_2yvals.append(calc_error(stm, transfer_time, x_0, r * m_2argmax))

print(m_0yvals)
print(m_1yvals)
print(m_2yvals)
fig, axs = plt.subplots(3, sharex=True)
axs[0].plot(xvals, m_0yvals)
axs[0].set_title("Method 0")
# Method 1 is baseline
axs[1].plot(xvals, m_1yvals)
axs[1].set_title("Method 1")
axs[2].plot(xvals, m_2yvals)
axs[2].set_title("Method 2")
plt.title("Error in Orbit Propagation vs Difference in Initial Perturbation")
plt.xlabel("Radius of Sphere of Perturbation")
plt.ylabel("Maximum Error")
plt.show()

