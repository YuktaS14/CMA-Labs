import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Length of the rod
L = 1
# mu = Thermal Diffusivity of the rod
mu = 0.01

# Given: Initial Condition => u(x,0) = e⁻ˣ and
#        Boundary Condition => u(0,t) = u(1,t) = 0
def initial_condition(x):
    ans = np.exp(-x)
    ans[0]=0
    ans[-1] = 0
    return ans

# assuming points on rod lie at distance of h = 0.01 from each other
h = 0.01
# considering time between 0 to 5
t0= 0
T = 5

# taking timestamps of temp for each point on the rod every 5/250th time
tls = np.linspace(t0,T,250)

# generating uniform points on the rod
x= np.arange(0,L+h,h)

# PDE for heat conduction in a unit length rod
def pde_heat_eq(t,u):
    dudt = np.zeros_like(u)
    # u(0,t) = 0 (boundary Condition)
    dudt[0] = 0

    # using central finite difference to find du/dt at each of the x point
    for i in range(1,len(u)-1):
        dudt[i] = mu*(u[i+1]-2*u[i]+u[i-1])/(h**2)
    
    # u(L,t) = 0 (boundary Condition)
    dudt[-1] = 0
    return dudt

# solving the PDE using initial condition
solvePDE = solve_ivp(pde_heat_eq, t_span=[t0,T], y0=initial_condition(x), t_eval=tls)

# retrieving values obtained for temperatures, after solving the PDE, at different x positions and at different time
temps= solvePDE.y
time = solvePDE.t

#Creating the animation
fig = plt.figure(figsize=(6, 4))

# update function for animation
def update(i):
    # clear the current figure to plot the next
    plt.clf()

    # plotting the temperature at time[i] for each of the x points
    atTimet = list(temps[r][i] for r in range(len(x)))
    plt.plot(x,atTimet,'r')
    #title of the plot
    plt.title(f"Temperatures at different \npositions of the rod at Time: {time[i]:.2f}")
    #setting x-y labels
    plt.xlabel("Position on Rod")
    plt.ylabel("Temperature")
    

# Create the animation
ani = FuncAnimation(fig, update, frames=len(time), interval=30)
# ani.save('q1.gif')
#Show the animation
plt.show()