import numpy as np
import scipy
from STMint.STMint import STMint
from STMint import TensorNormUtilities as tnu
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


def calc_error(stm, transfer_time, r_f, x_0, perturbation):
    delta_r_f_star = perturbation

    delta_v_0_1 = np.matmul(np.linalg.inv(stm[0:3, 3:6]), delta_r_f_star)

    y_0 = x_0 + np.array([0, 0, 0, *delta_v_0_1])

    r_f_1 = integrator.dynVar_int2([0, transfer_time], y_0, output="final")[0][:3]

    delta_r_f_1 = r_f_1 - r_f

    return np.linalg.norm(delta_r_f_star - delta_r_f_1, ord=2)


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
    inv_stm_rv = np.linalg.inv(stm_rv)
    return 0.5 * np.einsum("ilm,lj,mk->ijk", stt_rvv, inv_stm_rv, inv_stm_rv)


# Gateway Nrho ics: transfer sampling error
mu = 1.0 / (81.30059 + 1.0)
x0 = 1.02202151273581740824714855590570360
z0 = 0.182096761524240501132977765539282777
yd0 = -0.103256341062793815791764364248006121
period = 1.5111111111111111111111111111111111111111

x_0 = np.array([x0, 0, z0, 0, yd0, 0])

# Nrho Propagator
integrator = STMint(preset="threeBody", preset_mult=mu, variational_order=2)

transfer_time = period * 0.4

x_f, stm, stt = integrator.dynVar_int2([0, transfer_time], x_0, output="final")
r_f = x_f[:3]

s_0yvals = []
s_1yvals = []
s_2yvals = []
s_3yvals = []
m_1yvals = []
m_2yvals = []
m_3yvals = []
xvals = []

E1 = calc_e_tensor(stm, stt)

E1guess = np.array([1, 1, 1]) / np.linalg.norm(np.array([1, 1, 1]), ord=2)
tensSquared = np.einsum("ijk,ilm->jklm", E1, E1)
E1ArgMax, E1Norm = tnu.power_iteration_symmetrizing(tensSquared, E1guess, 100, 1e-9)

for i in range(0, 10):
    # Change so r is linearly distributed
    r = np.linalg.norm(x_0[3:]) / 100000.0 * ((i + 1) * 50.0)
    xvals.append(r)

    # Method 0: Sampling
    s_0yvals.append(
        calc_sphere_max_error(
            stm, transfer_time, r_f, x_0, normalize_sphere_samples(r, 100)
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
    m_1yvals.append(pow(r, 2) * np.sqrt(E1Norm))

    # Method 2: Making an educated guess at the maximum error.
    m_2yvals.append(calc_error(stm, transfer_time, r_f, x_0, r * E1ArgMax))

    # Method 3: Least Squares Error Maximization
    initial_guess = np.array([*(E1ArgMax * r)])

    err = lambda pert: calc_error(stm, transfer_time, r_f, x_0, pert)
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

# Change xvals' units to meters
# xvals_m = [x * 1000 for x in xvals]

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
axs[3].set_xlabel("Radius of Relative Final Position (km)", fontsize=16)
fig.text(
    0.06,
    0.5,
    "Maximum Error (km)",
    ha="center",
    va="center",
    rotation="vertical",
    fontsize=16,
)
plt.subplots_adjust(hspace=1, left=0.2, right=0.9)

# Plotting only method 3
fig2, model3 = plt.subplots(figsize=(8, 4.8))
model3.plot(xvals, m_3yvals)
model3.set_xlabel("Radius of Relative Final Position (km)", fontsize=16)
model3.set_ylabel("Maximum Error (km)", fontsize=16)

# Plotting error between methods (1 and 2 with resepct to 3)
error1_3 = []
error2_3 = []
for i in range(len(xvals)):
    error1_3.append((abs((m_1yvals[i] - m_3yvals[i])) / m_3yvals[i]) * 100)
    error2_3.append((abs((m_2yvals[i] - m_3yvals[i])) / m_3yvals[i]) * 100)

fig3, error = plt.subplots(figsize=(7, 4.8))
error.plot(xvals, error1_3, label="Methods 1 and 3")
error.plot(xvals, error2_3, label="Methods 2 and 3")
error.set_xlabel("Radius of Relative Final Position (km)", fontsize=16)
error.set_ylabel("Method Percentage Error", fontsize=16)
error.legend()

plt.show()
