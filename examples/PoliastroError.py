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
x_0 = np.array([*iss_orbit.r, *iss_orbit.v])


# Step 1
def calc_delta_v_0(integrator, transfer_time, x_0, delta_r_f_star):
    phi = integrator.dynVar_int2([0, transfer_time], x_0, output="final")[1]

    phi_rv = phi[:3, 3:]

    phi_rv_inv = np.linalg.inv(phi_rv)

    return np.matmul(phi_rv_inv, delta_r_f_star)

# Step 2

def calc_delta_r_f_error(integrator, transfer_time, x_0, delta_r_f_star):
    v_0 = calc_delta_v_0(integrator, transfer_time, x_0, delta_r_f_star)

    r_f = integrator.dynVar_int([0, transfer_time], x_0, output="final")[0][:3]

    delta_x_0 = x_0 + np.array([0, 0, 0, *v_0])

    delta_r_f = integrator.dynVar_int([0, transfer_time], delta_x_0, output="final")[0][:3] - r_f

    return np.linalg.norm((delta_r_f_star - delta_r_f), ord=2)

# Step 3

def calc_sphere_max_error(integrator, transfer_time, x_0, r, n):
    samples = np.random.multivariate_normal([0, 0, 0], np.identity(3), n)

    normalized_samples = []

    for sample in samples:
        normalized_samples.append((sample / np.linalg.norm(samples, ord=2)) * r)

    errors = []

    for sample in normalized_samples:
        errors.append(calc_delta_r_f_error(integrator, transfer_time, x_0, sample))

    return max(errors)

yvals = []
xvals = []
integrator = STMint(preset="twoBodyEarth", variational_order=2)

for i in range(100, 1000, 50):
    r = x_0[2] / (1100.0 - i)
    xvals.append(r)
    n = 500
    yvals.append(calc_sphere_max_error(integrator, iss_orbit.period/10.0, x_0, r, n))
    print(str((i - 50) / 50) + " completed")

plt.figure()
plt.plot(xvals, yvals)
plt.title("Error in Orbit Propagation vs Difference in Initial Perturbation")
plt.xlabel("Radius of Sphere of Perturbation")
plt.ylabel("Maximum Error")
plt.show()