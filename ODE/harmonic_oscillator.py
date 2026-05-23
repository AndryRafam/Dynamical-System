import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# harmonic oscillator frequency (s-1)
omega = 0.9
# initial conditions on x1 = x and x2 = dx/dt at t = 0
A, v0 = 3, 0 # cm, cm.s-1
x0 = A, v0

t0, tf = 0, 20

def dxdt(t,x):
	x1,x2 = x
	dx1dt = x2
	dx2dt = -omega**2*x1
	return [dx1dt, dx2dt]

# integrate the diff equation
soln = solve_ivp(dxdt,(t0,tf),x0,method="Radau",dense_output = True,t_eval=(t0,tf))

t = np.linspace(t0,tf,100)
x1,x2 = soln.sol(t)

plt.figure(facecolor="#dddddd",figsize=(12,8))
plt.plot(t,x1,'o',color='blue',label='solve ivp()')
plt.plot(t,A*np.cos(omega*t),color='red',label='Exact')
plt.xlabel(r'$t\;/\mathrm{s}$')
plt.ylabel(r'$x\;/\mathrm{cm}$')
plt.legend()
plt.savefig("harmonic_oscillator.png")
plt.show()