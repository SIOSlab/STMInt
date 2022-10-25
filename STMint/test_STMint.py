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
        self.testDynmaicsAndVariational1 = self.test1.dynVar_int([0,(2*math.pi)],
                                                    [1,0,0,0,1,0], max_step=.1)
        
        self.testDynmaicsAndVariational2 = self.test1.dynVar_int([0,(2*math.pi)],
                                                    [1,0,0,0,1.001,0], max_step=.1)

        difX = self.testDynmaicsAndVariational2.y[0][-1] - self.testDynmaicsAndVariational1.y[0][-1]
        difY = self.testDynmaicsAndVariational2.y[1][-1] - self.testDynmaicsAndVariational1.y[1][-1]
        difZ = self.testDynmaicsAndVariational2.y[2][-1] - self.testDynmaicsAndVariational1.y[2][-1]
        difVx = self.testDynmaicsAndVariational2.y[3][-1] - self.testDynmaicsAndVariational1.y[3][-1]
        difVy = self.testDynmaicsAndVariational2.y[4][-1] - self.testDynmaicsAndVariational1.y[4][-1]
        difVz = self.testDynmaicsAndVariational2.y[5][-1] - self.testDynmaicsAndVariational1.y[5][-1]

        t_f = []

        for i in range(len(self.testDynmaicsAndVariational1.y)):
            t_f.append(self.testDynmaicsAndVariational1.y[i][-1])

        phiT_f = Matrix(np.reshape(t_f[6:], (6,6)))

        IVPdeltaX_f = Matrix([difX,difY,difZ,difVx,difVy,difVz])

        NumericalDeltaX_f = phiT_f * Matrix([.001,0,0,0,0.001,0])


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
