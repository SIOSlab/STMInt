A tool for numerical integration of variational equations associated with a symbolically specified dynamical system.

STMint or State Transition Matrix Integrator uses sympy and scipy to symbolically construct variational equations and integrate them numerically.

STMint Installation
===================
---
If cloning from github, in the cloned STMint directory:

```
    pip install --user .
```

or, to install in developer mode:

```
    pip install --user -e .
```

---
**NOTE**

To upgrade to the latest version, just append ``--upgrade`` to whichever install command you originally used.  For example: ``pip install --upgrade --user STMint``.

Example Usage
===================
---
Suppose that we wish to propagate a satellite around the Earth. 
We achieve this goal by implementing the two-body dynamics symbolically using SymPy, and integrating using STMint.

To begin, we define the variables we will be using in this problem:

```
    x, y, z, vx, vy, vz = sympy.symbols("x, y, z, vx, vy, vz")
```

Next, we define our gravitational potential as V = (Gm/r), assuming that the Earth is stationary (mass of satellite can be ignored).
Note that we convert our cartesian coordinate to "r", and use the gravitational constant multiple from astropy: astropy.constants.GM_earth.

```
    V = astropy.constants.GM_earth / sympy.sqrt(x**2 + y**2 + z**2)
```

The desired units of this constant can be changed according to the documentation of astropy: https://docs.astropy.org/en/stable/constants/index.html, by default, this constant has (m^3/s^2)

We now define our symbolic radius matrix, and time derivative of the same:

```
    r = sympy.Matrix([x, y, z])
    vr = sympy.Matrix([vx, vy, vz])
```

Now, we take the gradient of our potential with respect to our symbolic radius to find our force.

```
    dVdr = sympy.diff(V, r)
```

Finally, we stack the right-hand side of our equations of motion, and pass these variables and dynamics into STMint to create an integrator with the given dynamics:

```
    RHS = sympy.Matrix.vstack(vr, dVdr)
    
    integrator = STMint(vars=sympy.Matrix(x, y, z, vx, vy, vz), dynamics=RHS)
```

Suppose the satellite we wish to propagate is the ISS, we define it's initial conditions as follows:

```
    # ISS Keplerian Elements
    a = 6738000 << u.m
    ecc = 0.0005140 << u.one
    inc = 51.6434 << u.deg
    
    # Since these elements are constantly changing, they will be ignored for simplicity
    raan = 0 << u.deg
    argp = 0 << u.deg
    nu = 0 << u.deg
```

We can use the package poliastro to convert these Keplerian elements to a 6 cartesian initial conditions and find the satellite's period.

```
    iss_orbit = poliastro.twobody.orbit.from_classical(poliastro.bodies.Earth, a, ecc, inc, raan, argp, nu)
    
    period = iss_orbit.period.to(u.s).value
    
    x_0 = np.array([*iss_orbit.r.value, *iss_orbit.v.value])
```

We now can preform our desired propagation, let suppose we wish to find the position of the satellite after 1/2 period.

```
    final_state, final_stm = integrator.dynVar_int([0, period / 2.0], x_0, output="final")
```

In the above example, final_state is our position, and final_stm is the numerical state transition matrix at this position.

What if we would like to plot the path that this satellite takes? We can do so by changing our output from "final" to "all":

```
    all_states, all_stms, all_ts = integrator.dynVar_int([0, period / 2.0], x_0, output="all")
```

In this case, all_ts provides each time step STMint used for numerical integration of the provided dynamics. By plotting these states against each time step, we are able to create a graph of the desired trajectory.
The length of these time steps may be customized aswell.

---
**NOTE**

This example follows the workflow provided in the preset "twoBodyEarth", and the integration steps in TwoBodyPropError.py.
There also exists presets for a two body system with the sun, and multiple restricted three-body problems.

STMint Documentation
====================
Documentation is available here: https://stmint.readthedocs.io/