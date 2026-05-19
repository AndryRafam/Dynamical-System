import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# define the ODE of the damped harmonic oscillator
def dy(t,y,zeta,w0):
	x,p = y
	dx = p
	dp = -2*zeta*w0*p-w0**2*x
	return np.array([dx,dp])

# initial state:
y0 = [1.0,0.0]

# time coordinate to solve the ODE for
t = np.linspace(0,10,1000)
w0 = 2*np.pi*1.0

zeta = [0.0,0.2,1.0,5.0]
sol = [0 for i in range(4)]
color = ["black","red","blue","green"]
label = ["undamped","under damped","critical damping","over damped"]

fig, ax = plt.subplots(figsize=(12,8),facecolor="#dddddd")

for i in range(4):
	sol[i] = solve_ivp(dy,(0,10),y0,method="RK23",args=(zeta[i],w0))
	ax.plot(sol[i].t,sol[i].y[0],color=color[i],label=label[i],lw=2)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend()

plt.savefig("damped_harmonic_oscillator.png")
plt.show()