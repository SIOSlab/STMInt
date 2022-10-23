from sympy import *
import numpy as np
from scipy.integrate import solve_ivp
import math

class STMint:
    """ State Transition Matrix Integrator
    A tool for numerical integration of variational
    equations associated with a symbolically specified dynamical system.
    """

    # MUTABLE ATTRIBUTES
    # Attribute vars: the variables used in the symbolic integration
    # Invariant: vars is a list
    #
    # Attribute dynamics: the dynamics to be symbolically integrated
    # Invariant: dynamics is the dynamic equation(s) in the form
    # of sympy expressions
    #
    # Attribute lambda_dynamics: the lambdified dynamic equations
    # Invariant: lambda_dynamics is a lambdified sympy expression
    #
    # Attribute jacobian: the jacobian of the dynamics
    # Invariant: jacobian is an n-dimensional sympy Matrix
    #
    # Attribute STM: the symbolic state transition matrix
    # Invariant: STM is an nxn sympy Matrix, the dimensions of STM is equal to
    # number of variables used
    #
    # Attribute variational: the variatonal equations of the dynamical system
    # Invariant: variational is an n-dimensional sympy Matrix
    #
    # Attribute lambda_dynamics_and_variational: the lambdified dynamic and
    # variational equations
    # Invariant: lambda_dynamics_and_variational is a lambdified sympy expression

    def __init__(self, v=None, dyn=None, preset="", variation=True):

        # preset for two body motion
        if(preset=="twoBody"):

            # two body motion dynmaics
            x,y,z,vx,vy,vz=symbols("x,y,z,vx,vy,vz")
            V = 1/sqrt(x**2+y**2+z**2)
            r = Matrix([x,y,z])
            vr = Matrix([vx,vy,vz])
            dVdr = diff(V,r)
            left = Matrix.vstack(r,vr)

            self.vars = Matrix([x,y,z,vx,vy,vz])
            self.dynamics = left
        else:
            for elem in v:
                elem=symbols(str(elem))

            self.vars = Matrix(v)
            self.dynamics = dyn

        self.lambda_dynamics = lambdify(self.vars, self.dynamics, "numpy")

        # if user wants to use variational equations
        if(variation):
            self.jacobian = self.dynamics.jacobian(self.vars.transpose())
            self.STM = MatrixSymbol("phi",len(self.vars),len(self.vars))
            self.variational = self.jacobian * self.STM
            self.lambda_dynamics_and_variational = lambdify((self.vars,self.STM),
                                        Matrix.vstack(self.dynamics.transpose(),
                                        Matrix(self.variational)), "numpy")


# =============================================================================
# IVP Functions
# =============================================================================


    def _dynamics_solver(self, t, y):
        """Function to mimic right hand side of a dynamic system for integration

        Method unpacks initial coniditions y from solve_ivp and sends it to the
        predefined lamdified dynamics.

        Args :
            t (float)
                Independent variable of initial conditions

            y (float n array)
                Array of initial conditions of solve_ivp

        Returns:
            lambda_dynamics (float n array)
                Array of values of dynamics subjected to initial conditions
        """

        lambda_dynamics = self.lambda_dynamics(*y).flatten()

        return lambda_dynamics


    def _dynamics_and_variational_solver(self, t, y):
        """Function to mimic right hand side of a dynamic system with variational
            equations integrattion

        Method unpacks initial coniditions y from solve_ivp and sends it to the
        predefined lamdified dynamics and variational equations.

        Args :
            t (float)
                Independent variable of initial conditions

            y (float n array)
                Array of initial conditions of solve_ivp

        Returns:
            lambda_dynamics_and_variational (float n array)
                Array of values of dynamics and variational equations subjected
                to initial conditions
        """
        
        l = len(self.vars)
        lambda_dynamics_and_variational = self.lambda_dynamics_and_variational(y[:l],
                                    np.reshape(y[l:], (l,l))).flatten()

        return lambda_dynamics_and_variational


# =============================================================================
# Clones of solve_ivp
# =============================================================================


    def dyn_int(self, t_span, y0, method='RK45', t_eval=None,
                            dense_output=False, events=None, vectorized=False,
                            args=None, **options):

        return solve_ivp(self._dynamics_solver, t_span, y0, method, t_eval,
                        dense_output, events, vectorized, args, **options)


    def dynVar_int(self, t_span, y0, method='RK45', t_eval=None,
                            dense_output=False, events=None, vectorized=False,
                            args=None, **options):

        initCon = np.vstack((np.array(y0),np.eye(len(self.vars))))

        return solve_ivp(self._dynamics_and_variational_solver, t_span,
            initCon.flatten(), method, t_eval, dense_output, events, vectorized, args, **options)
