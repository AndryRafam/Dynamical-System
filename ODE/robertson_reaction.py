import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def deriv(t,y):
	"""
	ODEs for Robertson's chemical reaction system.
	"""
	x,y,z = y
	xdot = -0.04 * x + 1.e4 * y * z
	ydot = 0.04 * x - 1.e4 * y * z - 3.e7 * y**2
	zdot = 3.e7 * y**2
	return [xdot,ydot,zdot]

# initial and final times.
t0, tf = 0, 500
# initial conditions: [X] = 1; [Y] = [Z] = 0.
y0 = 1, 0, 0
# solve, using a method resilient to stiff ODEs.
soln = solve_ivp(deriv,(t0,tf),y0,method="Radau")
print(soln.nfev, 'evaluations required.')

YFAC = 4
plt.figure(facecolor="#dddddd",figsize=(12,8))
plt.plot(soln.t, soln.y[0], lw=2, label='[X]')
plt.plot(soln.t, 10**YFAC*soln.y[1], lw=2, label=r'$10^{}\times$[Y]'.format(YFAC))
plt.plot(soln.t, soln.y[2], lw=2, label='[Z]')
plt.xlabel('time /s')
plt.ylabel('concentration /arb. units')
plt.legend()
plt.grid(True)
plt.savefig("robertson_reaction.png")
plt.show()
