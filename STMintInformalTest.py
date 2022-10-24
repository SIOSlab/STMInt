from sympy import *
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math
from STMint import STMint

x,y,z,vx,vy,vz=symbols("x,y,z,vx,vy,vz")

V = 1/sqrt(x**2+y**2+z**2)
r = Matrix([x,y,z])
vr = Matrix([vx,vy,vz])

dVdr = diff(V,r)

dynamics = Matrix.vstack(vr,dVdr)

mySTMint = STMint([x,y,z,vx,vy,vz], dynamics)

solf = mySTMint.dyn_int([0,(2*math.pi)], [1,0,0,0,1,0], max_step=.1)

sol1 = mySTMint.dynVar_int([0,(2*math.pi)], [1,0,0,0,1,0], max_step=.1)

mySTMint1 = STMint(preset="twoBody")

sol2 = mySTMint1.dynVar_int([0,(2*math.pi)], [1,0,0,0,1,0], max_step=.1)

t_f = []

for i in range(len(sol1.y)):
    t_f.append(sol1.y[i][-1])

phiT_f = Matrix(np.reshape(t_f[6:], (6,6)))
#print(shape(sol1.y))
#print(sol1.y)

# Plotting
ax = plt.figure().add_subplot(projection='3d')

ax.plot(sol1.y[0],sol1.y[1],sol1.y[2])

ax1 = plt.figure().add_subplot(projection='3d')

ax1.plot(sol2.y[0],sol2.y[1],sol2.y[2])

ax2 = plt.figure().add_subplot(projection='3d')

ax2.plot(solf.y[0],solf.y[1],solf.y[2])

plt.show()
