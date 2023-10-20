A tool for numerical integration of variational equations associated with a symbolically specified dynamical system.

STMint or State Transition Matrix Integrator uses sympy and scipy to symbolically construct variational equations and integrate them numerically.

STMint Installation
===================
To install from PyPI:

```
    pip install --user STMint
```

To install system-wide, omit the ``--user`` option. This requires administrative privileges on most systems.

---
If cloning from github, in the cloned grading directory:

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

STMint Documentation
====================
Documentation is available here: https://stmint.readthedocs.io/