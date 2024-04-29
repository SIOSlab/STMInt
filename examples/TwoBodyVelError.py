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
        integrator,
        x_0[:3],
        (delta_v_0_1 + x_0[3:]),
        (r_f + delta_r_f_star),
        transfer_time,
        stm,
        10e-12,
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

    E = 0.5 * np.einsum("ilm,lj,mk->ijk", stt_rvv, inv_stm_rv, inv_stm_rv)
    tensSquared = np.einsum("ijk,ilm->jklm", E, E)
    ENormMax = 0
    EArgMaxMax = 0
    # try 10 different initial guesses for symmetric higher order power iteration
    for i in range(10):
        Eguess = np.random.multivariate_normal([0, 0, 0], np.identity(3), 1)[0]
        Eguess = Eguess / np.linalg.norm(Eguess, ord=2)
        EArgMax, ENorm = tnu.power_iteration_symmetrizing(
            tensSquared, Eguess, 100, 1e-9
        )
        if ENorm > ENormMax:
            ENormMax = ENorm
            EArgMaxMax = EArgMax
    return E, EArgMaxMax, ENormMax


def calc_e_prime_tensor(e_tens, stm):
    stm_rv = stm[0:3, 3:6]
    inv_stm_rv = np.linalg.inv(stm_rv)

    Eprime = np.einsum("il,ljk->ijk", inv_stm_rv, e_tens)
    EprimeTensSquared = np.einsum("ijk,ilm->jklm", Eprime, Eprime)
    EprimeNormMax = 0
    EprimeArgMaxMax = 0

    for i in range(10):
        EprimeGuess = np.random.multivariate_normal([0, 0, 0], np.identity(3), 1)[0]
        EprimeGuess = EprimeGuess / np.linalg.norm(EprimeGuess, ord=2)
        EprimeArgMax, EprimeNorm = tnu.power_iteration_symmetrizing(
            EprimeTensSquared, EprimeGuess, 100, 1e-9
        )
        if EprimeNorm > EprimeNormMax:
            EprimeNormMax = EprimeNorm
            EprimeArgMaxMax = EprimeArgMax

    return Eprime, EprimeArgMaxMax, EprimeNormMax


def newton_root_velocity(
    integrator, r_0, v_n, r_f, transfer_time, stm_n, tolerance, termination_limit=100
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
    if termination_limit == 0:
        return v_n
    elif np.linalg.norm(residual) <= tolerance:
        return v_n
    else:
        delta_v_0_n = np.matmul(np.linalg.inv(stm_n[0:3, 3:6]), residual)

        v_0_n_1 = v_n - delta_v_0_n

        return newton_root_velocity(
            integrator,
            r_0,
            v_0_n_1,
            r_f,
            transfer_time,
            stm_n,
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

period = iss_orbit.period.to(u.s).value
transfer_time = period * 0.1

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
_, stm, stt = integrator.dynVar_int2([0, transfer_time], x_0, output="final")

E1 = calc_e_tensor(stm, stt)[0]
E1prime, E1primeArgMax, E1primeNorm = calc_e_prime_tensor(E1, stm)

# Tensor Norm Calculations

tensor_norms = []

_, stms, stts, ts = integrator.dynVar_int2(
    [0, period], x_0, max_step=(transfer_time) / 100.0, output="all"
)

for i in range(1, len(ts)):
    tensor_norms.append(
        calc_e_prime_tensor(calc_e_tensor(stms[i], stts[i])[0], stms[i])[2]
    )


for i in range(0, 20):
    # Scale of 200km
    r = 10 * (i + 1)
    xvals.append(r)

    # Method 0: Sampling
    s_0yvals.append(
        calc_sphere_max_error(
            integrator, stm, transfer_time, r_f, x_0, normalize_sphere_samples(r, 5000)
        )
    )

    # Method 1: Analytical method for calculating maximum error
    m_1yvals.append(pow(r, 2) * np.sqrt(E1primeNorm))

    # Method 2: Making an educated guess at the maximum error
    err_eval1 = calc_error(integrator, stm, transfer_time, r_f, x_0, r * E1primeArgMax)
    err_eval2 = calc_error(
        integrator, stm, transfer_time, r_f, x_0, -1.0 * r * E1primeArgMax
    )
    m_2yvals.append(max(err_eval1, err_eval2))

    # Method 3: Least Squares Error Maximization
    if err_eval1 > err_eval2:
        initial_guess = np.array([*(E1primeArgMax * r)])
    else:
        initial_guess = np.array([*(-1.0 * E1primeArgMax * r)])

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
        tol=(1e-12),
        options={"disp": True},
    )

    m_3yvals.append(err(min.x))

# Changing seconds to periods
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
axs[3].set_xlabel("Radius of Relative Final Position (km)", fontsize=16)
fig.text(
    0.06,
    0.5,
    "Maximum Error (km/s)",
    ha="center",
    va="center",
    rotation="vertical",
    fontsize=16,
)
plt.subplots_adjust(hspace=1, left=0.2, right=0.9)

# Plotting only method 3
fig2, model3 = plt.subplots(figsize=(8, 6))
model3.plot(xvals, np.array(m_3yvals) * 1e6, linewidth=4)
model3.set_xlabel("Radius of Relative Final Position (km)", fontsize=18)
model3.set_ylabel("Maximum Error (mm/s)", fontsize=18)
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

# Error 2_3 below 10^-7 level


error.set_xlabel("Radius of Relative Final Position (km)", fontsize=18)
error.set_ylabel("Method Percentage Error", fontsize=18)
error.set_yscale("log")
error.legend(fontsize=14)
error.tick_params(labelsize=14)

xvals = np.array([0, 0.25, 0.5, 0.75, 1])
xlabels = ["0", "1/4", "1/2", "3/4", "1"]

fig4, norms = plt.subplots(figsize=(8, 6))
norms.plot(ts[21:], tensor_norms[20:], linewidth=4)
norms.set_xlabel("Time of Flight (periods)", fontsize=18)
norms.set_ylabel("Tensor Norm (km s)^-1", fontsize=18)
norms.set_yscale("log")
norms.tick_params(labelsize=14)
norms.set_xticks(xvals, xlabels, fontsize=14)

fig2.savefig("figures/Vel/twoBodyVelOpt.png")
fig3.savefig("figures/Vel/twoBodyVelError.png")
fig4.savefig("figures/Vel/twoBodyVelTNorms.png")
