from sympy import *
import numpy as np
from astropy import units as u
from STMint.STMint import STMint
from poliastro.twobody.orbit import Orbit
import poliastro.bodies as body
import matplotlib.pyplot as plt

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

    print((r_f_ref - r_f_pert))
    print(np.matmul(stm, delta_x_0)[:3])
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

    return max(errors)

yvals = []
xvals = []
integrator = STMint(preset="twoBodyEarth")
transfer_time = iss_orbit.period.to(u.s).value/10.0
stm = integrator.dynVar_int([0, transfer_time], x_0, output="final")[1]

for i in range(100, 1000, 50):
    r = np.linalg.norm(x_0[3:]) / (1100.0 - i)
    xvals.append(r)
    n = 500
    yvals.append(calc_sphere_max_error(stm, transfer_time, x_0, r, n))
    print(str((i - 50) / 50) + " completed")

plt.figure()
plt.plot(xvals, yvals)
plt.title("Error in Orbit Propagation vs Difference in Initial Perturbation")
plt.xlabel("Radius of Sphere of Perturbation")
plt.ylabel("Maximum Error")
plt.show()