import unittest
from sympy import *
import numpy as np
from STMint.STMint import STMint
from util import skew, findSTM
import math
from STMint import TensorNormUtilities as tnu


class TestTwoBodyMotion(unittest.TestCase):
    def setUp(self):
        x, y, z, vx, vy, vz = symbols("x,y,z,vx,vy,vz")

        V = 1 / sqrt(x**2 + y**2 + z**2)
        r = Matrix([x, y, z])
        vr = Matrix([vx, vy, vz])

        dVdr = diff(V, r)

        dynamics = Matrix.vstack(vr, dVdr)

        self.test1 = STMint([x, y, z, vx, vy, vz], dynamics)

        self.testDynamicsAndVariational1 = self.test1.dynVar_int(
            [0, (2 * math.pi)], [1, 0, 0, 0, 1, 0], max_step=0.1
        )

    def test_dyn_int(self):
        self.testDynamics = self.test1.dyn_int(
            [0, (2 * math.pi)], [1, 0, 0, 0, 1, 0], max_step=0.1
        )

    def test_preset(self):
        self.presetTest = STMint(preset="twoBody")

        self.testDynamicsAndVariational2 = self.presetTest.dynVar_int(
            [0, (2 * math.pi)], [1, 0, 0, 0, 1, 0], max_step=0.1
        )

    def test_dynVar_int(self):
        self.testDynamicsAndVariational2 = self.test1.dynVar_int(
            [0, (2 * math.pi)], [1, 0, 0, 0, 1, 0], output="raw", max_step=0.1
        )

        self.testDynamicsAndVariational3 = self.test1.dynVar_int(
            [0, (2 * math.pi)], [1, 0, 0, 0, 1, 0], output="final", max_step=0.1
        )

        self.testDynamicsAndVariational4 = self.test1.dynVar_int(
            [0, (2 * math.pi)], [1, 0, 0, 0, 1, 0], output="all", max_step=0.1
        )

    def test_dynVar_int2(self):
        self.presetTest = STMint(preset="twoBody", variational_order=2)

        self.testDynamicsAndVariational2_10 = self.presetTest.dynVar_int2(
            [0, (2 * math.pi)], [1, 0, 0, 0, 1, 0], output="final", max_step=0.001
        )
        # run the nonlinearity index calculation
        nonlinearity_index = tnu.nonlin_index_2_eigenvector(
            self.testDynamicsAndVariational2_10[1],
            self.testDynamicsAndVariational2_10[2],
        )

        # find second leg of stm and stt
        self.testDynamicsAndVariational2_21 = self.presetTest.dynVar_int2(
            [2 * math.pi, 3 * math.pi],
            [1, 0, 0, 0, 1, 0],
            output="final",
            max_step=0.001,
        )
        # find stm and stt along total interval
        self.testDynamicsAndVariational2_20 = self.presetTest.dynVar_int2(
            [0, 3 * math.pi], [1, 0, 0, 0, 1, 0], output="final", max_step=0.001
        )
        stt2_reference = self.testDynamicsAndVariational2_20[2]
        stt2_cocycle = tnu.cocycle2(
            self.testDynamicsAndVariational2_10[1],
            self.testDynamicsAndVariational2_10[2],
            self.testDynamicsAndVariational2_21[1],
            self.testDynamicsAndVariational2_21[2],
        )[1]
        self.assertTrue(np.amax(np.abs((stt2_cocycle - stt2_reference))) < 1e-8)

    def test_propagation(self):
        self.testDynamicsAndVariational2 = self.test1.dynVar_int(
            [0, (2 * math.pi)], [1, 0, 0, 0, 1.001, 0], max_step=0.1
        )

        difX = (
            self.testDynamicsAndVariational2.y[0][-1]
            - self.testDynamicsAndVariational1.y[0][-1]
        )
        difY = (
            self.testDynamicsAndVariational2.y[1][-1]
            - self.testDynamicsAndVariational1.y[1][-1]
        )
        difZ = (
            self.testDynamicsAndVariational2.y[2][-1]
            - self.testDynamicsAndVariational1.y[2][-1]
        )
        difVx = (
            self.testDynamicsAndVariational2.y[3][-1]
            - self.testDynamicsAndVariational1.y[3][-1]
        )
        difVy = (
            self.testDynamicsAndVariational2.y[4][-1]
            - self.testDynamicsAndVariational1.y[4][-1]
        )
        difVz = (
            self.testDynamicsAndVariational2.y[5][-1]
            - self.testDynamicsAndVariational1.y[5][-1]
        )

        # Basic Numerical Calculation
        t_f = []

        for i in range(len(self.testDynamicsAndVariational1.y)):
            t_f.append(self.testDynamicsAndVariational1.y[i][-1])

        phiT_f = Matrix(np.reshape(t_f[6:], (6, 6)))

        IVPdeltaX_f = Matrix([difX, difY, difZ, difVx, difVy, difVz])

        NumericalDeltaX_f = phiT_f * Matrix([0, 0, 0, 0, 0.001, 0])

        self.assertTrue(
            (((NumericalDeltaX_f - IVPdeltaX_f).norm()) / NumericalDeltaX_f.norm())
            < 0.02
        )

    def test_propagationWithFindSTM(self):
        # Calculating STM from findSTM
        finalPos = np.array(
            [
                self.testDynamicsAndVariational1.y[0][-1],
                self.testDynamicsAndVariational1.y[1][-1],
                self.testDynamicsAndVariational1.y[2][-1],
            ]
        )

        finalVel = np.array(
            [
                self.testDynamicsAndVariational1.y[3][-1],
                self.testDynamicsAndVariational1.y[4][-1],
                self.testDynamicsAndVariational1.y[5][-1],
            ]
        )

        stmUtil = findSTM(
            np.array([1, 0, 0]), np.array([0, 1, 0]), finalPos, finalVel, (2 * math.pi)
        )

        # Calculating STM from STMint
        t_f = []

        for i in range(len(self.testDynamicsAndVariational1.y)):
            t_f.append(self.testDynamicsAndVariational1.y[i][-1])

        stmSTMint = np.reshape(t_f[6:], (6, 6))
        self.assertTrue(
            (np.linalg.norm(stmUtil - stmSTMint) / np.linalg.norm(stmUtil)) < 1e-8
        )


class TestThreeBodyMotion(unittest.TestCase):
    def test_preset(self):
        self.presetTest = STMint(preset="threeBody")


class TestGeneralDynamics(unittest.TestCase):
    def setUp(self):
        a, b, c, d = symbols("a,b,c,d")
        A = Matrix([[a, b], [c, d]])
        self.simpleRight = Matrix([[1, 0], [0, 0]]) * A
        self.test1 = STMint([a, b, c, d], self.simpleRight, variational_order=0)

    def test_dyn_int(self):
        testDynamics = self.test1.dyn_int([0, 2], [1, 1, 1, 1], max_step=0.1)


if __name__ == "__main__":
    unittest.main()
