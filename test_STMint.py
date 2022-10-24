import unittest
from sympy import *
import numpy as np
from STMint import STMint
import math

class TestTwoBodyMotion(unittest.TestCase):
    def setUp(self):
        x,y,z,vx,vy,vz=symbols("x,y,z,vx,vy,vz")

        V = 1/sqrt(x**2+y**2+z**2)
        r = Matrix([x,y,z])
        vr = Matrix([vx,vy,vz])

        dVdr = diff(V,r)

        dynamics = Matrix.vstack(vr,dVdr)

        self.test1 = STMint([x,y,z,vx,vy,vz], dynamics)

    def test_dyn_int(self):
        self.testDynamics = self.test1.dyn_int([0,(2*math.pi)], [1,0,0,0,1,0], max_step=.1)

    def test_dynVar_int(self):
        self.testDynmaicsAndVariational1 = self.test1.dynVar_int([0,(2*math.pi)],
                                                    [1,0,0,0,1,0], max_step=.1)

    def test_propogation(self):
        self.testDynmaicsAndVariational2 = self.test1.dynVar_int([0,(2*math.pi)],
                                                    [1,0,0,0,1.001,0], max_step=.1)

        difX = sol2.y[0][-1] - sol1.y[0][-1]
        difY = sol2.y[1][-1] - sol1.y[1][-1]
        difZ = sol2.y[2][-1] - sol1.y[2][-1]
        difVx = sol2.y[3][-1] - sol1.y[3][-1]
        difVy = sol2.y[4][-1] - sol1.y[4][-1]
        difVz = sol2.y[5][-1] - sol1.y[5][-1]

        IVPdeltaX_f = Matrix([difX,difY,difZ,difVx,difVy,difVz])

        NumericalDeltaX_f = phiT_f * Matrix(deltaX_0)


class TestGeneralDynamics(unittest.TestCase):
    def setUp(self):
        a,b,c,d=symbols("a,b,c,d")
        A = Matrix([[a,b],[c,d]])
        self.simpleRight = Matrix([[1,0],[0,0]]) * A
        self.test1 = STMint([a,b,c,d], self.simpleRight, variation=False)

    def test_dyn_int(self):
        testDynamics = self.test1.dyn_int([0,2], [1,1,1,1], max_step = .1)



if __name__ == '__main__':
    unittest.main()
