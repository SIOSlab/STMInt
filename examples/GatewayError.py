import numpy as np
from STMint.STMint import STMint
import matplotlib.pyplot as plt

# Gateway Nrho ics: transfer sampling error
mu = 1.0 / (81.30059 + 1.0)
x0 = 1.02202151273581740824714855590570360
z0 = 0.182096761524240501132977765539282777
yd0 = -0.103256341062793815791764364248006121
period = 1.5111111111111111111111111111111111111111

x_0 = np.array([x0, 0, z0, 0, yd0, 0])


def calc_delta_v_0(integrator, transfer_time, x_0, delta_r_f_star):
    phi = integrator.dynVar_int([0, transfer_time], x_0, output="final")[1]

    phi_rv = phi[:3, 3:]

    phi_rv_inv = np.linalg.inv(phi_rv)

    return np.matmul(phi_rv_inv, delta_r_f_star)


def calc_delta_r_f_error(integrator, transfer_time, x_0, delta_r_f_star):
    v_0 = calc_delta_v_0(integrator, transfer_time, x_0, delta_r_f_star)

    r_f = integrator.dynVar_int([0, transfer_time], x_0, output="final")[0][:3]

    delta_x_0 = x_0 + np.array([0, 0, 0, *v_0])

    delta_r_f = (
        integrator.dynVar_int([0, transfer_time], delta_x_0, output="final")[0][:3]
        - r_f
    )

    return np.linalg.norm((delta_r_f_star - delta_r_f), ord=2)


def calc_sphere_max_error(integrator, transfer_time, x_0, r, n):
    samples = np.random.multivariate_normal([0, 0, 0], np.identity(3), n)

    normalized_samples = []

    for sample in samples:
        normalized_samples.append((sample / np.linalg.norm(sample, ord=2)) * r)

    errors = []

    for sample in normalized_samples:
        errors.append(calc_delta_r_f_error(integrator, transfer_time, x_0, sample))

    return max(errors)


yvals = []
xvals = []
integrator = STMint(preset="threeBody", preset_mult=1.0 / (81.30059 + 1.0))
num_of_radii = 20

for i in range(0, num_of_radii):
    r = np.linalg.norm(x_0[3:]) / 100000.0 * ((i + 1) * 50.0)
    xvals.append(r)
    n = 500
    yvals.append(calc_sphere_max_error(integrator, period / 10.0, x_0, r, n))
    print(str(i + 1) + "/" + str(num_of_radii) + " completed")

# Changing units to meters
xvals_m = [x * 1000 for x in xvals]
yvals_m = [x * 1000 for x in yvals]

plt.figure()
plt.plot(xvals_m, yvals_m)
plt.title("Error in Orbit Propagation vs Difference in Initial Perturbation")
plt.xlabel("Radius of Sphere of Perturbation (m/s)")
plt.ylabel("Maximum Error (m)")
plt.show()
