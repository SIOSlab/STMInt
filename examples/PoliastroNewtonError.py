import numpy as np
import scipy
from astropy import units as u
from STMint.STMint import STMint
from STMint import TensorNormUtilities as tnu
from poliastro.twobody.orbit import Orbit
import poliastro.bodies as body
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


def calc_error(integrator, stm, transfer_time, r_f, x_0, perturbation):
    delta_r_f_star = perturbation

    delta_v_0_1 = np.matmul(np.linalg.inv(stm[0:3, 3:6]), delta_r_f_star)

    v_0_newton = newton_root_velocity(
        integrator, x_0[:3], delta_v_0_1, (r_f + delta_r_f_star), transfer_time, 10e-12
    )

    return np.linalg.norm((x_0[3:] + delta_v_0_1) - v_0_newton, ord=2)


def normalize_sphere_samples(r, n):
    samples = np.random.multivariate_normal([0, 0, 0], np.identity(3), n)

    normalized_samples = []

    for sample in samples:
        normalized_samples.append((sample / np.linalg.norm(sample, ord=2)) * r)

    return normalized_samples


def calc_sphere_max_error(integrator, stm, transfer_time, r_f, x_0, normalized_samples):
    errors = []

    for sample in normalized_samples:
        errors.append(calc_error(integrator, stm, transfer_time, r_f, x_0, sample))

    return np.max(errors)


def calc_e_tensor(stm, stt):
    stm_rv = stm[0:3, 3:6]
    stt_rvv = stt[0:3, 3:6, 3:6]
    inv_stm_rv = np.linalg.inv(stm_rv)
    return 0.5 * np.einsum("ilm,lj,mk->ijk", stt_rvv, inv_stm_rv, inv_stm_rv)


def calc_f_tensor(e_tens, stm):
    stm_rv = stm[0:3, 3:6]
    return np.einsum("il,ljk->ijk", stm_rv, e_tens)


def newton_root_velocity(
    integrator, r_0, v_n, r_f, transfer_time, tolerance, termination_limit=10
):
    x_0_guess = np.hstack((r_0, v_n))
    r_f_n = (
        (
            Orbit.from_vectors(
                body.Earth, x_0_guess[:3] * u.km, x_0_guess[3:] * u.km / u.s
            )
        )
        .propagate(transfer_time * u.s)
        .r.value
    )

    residual = r_f_n - r_f
    print(np.linalg.norm(residual))
    if termination_limit == 0:
        return v_n
    elif np.linalg.norm(residual) <= tolerance:
        print("Tol after " + str(100 - termination_limit) + " iterations.")
        return v_n
    else:
        stm_n = integrator.dynVar_int([0, transfer_time], x_0_guess, output="final")[1]

        delta_v_0_n = np.matmul(np.linalg.inv(stm_n[0:3, 3:6]), residual)

        v_0_n_1 = v_n - delta_v_0_n

        return newton_root_velocity(
            integrator,
            r_0,
            v_0_n_1,
            r_f,
            transfer_time,
            tolerance,
            termination_limit - 1,
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

transfer_time = iss_orbit.period.to(u.s).value * 0.4

iss_reference_orbit = iss_orbit.propagate(transfer_time * u.s)

r_f = np.array([*iss_reference_orbit.r.value])

s_0yvals = []
s_1yvals = []
s_2yvals = []
s_3yvals = []
m_1yvals = []
m_2yvals = []
m_3yvals = []
xvals = []

integrator = STMint(preset="twoBodyEarth", variational_order=2)
stm = integrator.dynVar_int([0, transfer_time], x_0, output="final")[1]
stt = integrator.dynVar_int2([0, transfer_time], x_0, output="final")[2]

E1 = calc_e_tensor(stm, stt)
F1 = calc_f_tensor(E1, stm)

E1guess = np.array([1, 1, 1]) / np.linalg.norm(np.array([1, 1, 1]), ord=2)
EtensSquared = np.einsum("ijk,ilm->jklm", E1, E1)
E1ArgMax, E1Norm = tnu.power_iteration_symmetrizing(EtensSquared, E1guess, 100, 1e-9)

F1guess = np.array([1, 1, 1]) / np.linalg.norm(np.array([1, 1, 1]), ord=2)
FtensSquared = np.einsum("ijk,ilm->jklm", F1, F1)
F1ArgMax, F1Norm = tnu.power_iteration_symmetrizing(FtensSquared, F1guess, 100, 1e-9)

for i in range(0, 25):
    r = np.linalg.norm(x_0[3:]) / (100000) * ((i + 1) * 50)
    xvals.append(r)

    # Sampling Method with different number of samples.
    s_0yvals.append(
        calc_sphere_max_error(
            integrator, stm, transfer_time, r_f, x_0, normalize_sphere_samples(r, 10)
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
    m_1yvals.append(pow(r, 2) * np.sqrt(F1Norm))

    # Method 2: Making an educated guess at the maximum error.
    m_2yvals.append(calc_error(integrator, stm, transfer_time, r_f, x_0, r * F1ArgMax))

    # Method 3: Least Squares Error Maximization
    initial_guess = np.array([*(F1ArgMax * r)])

    err = lambda pert: calc_error(integrator, stm, transfer_time, r_f, x_0, pert)
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
        options={"ftol": 1e-9, "disp": True},
    )

    m_3yvals.append(err(min.x))
    print("Number of radii sampled: " + str(i + 1))

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
    "Maximum Error (m/s)",
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
model3.set_ylabel("Maximum Error (m/s)", fontsize=16)


# Plotting error between methods (1 and 2 with resepct to 3)
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