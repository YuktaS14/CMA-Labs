import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
import math

"""
    Function to find solution to the ODE of simple gravity pendulum
"""
def solveODE(t0,T,g,L,omega0,theta0,h):

    #d²(theta(t))/(d(t²))
    def ode2(omega,theta,t):
        return -(g/L)*math.sin(theta)

    # getting the x, y coordinates from value of theta
    def get_coords(theta):
        return L*math.sin(theta), -L*math.cos(theta)
    
    thetals = [theta0]
    omegals = [omega0]
    # generating t points with step size of h
    tls = list(np.arange(t0,T+h,h))

    # computing the values theta and omega from t, using forward Euler Method
    # solving 2 one ordered ODE : 1. d(theta)/dt = omega and 2. d(omega)/dt = -(g/L)*sin(theta)

    for i in range(len(tls)-1):
        omegaNext = omegals[i] + h*ode2(omegals[i],thetals[i],tls[i])
        thetaNext = thetals[i] + h*omegals[i]
        thetals.append(thetaNext)
        omegals.append(omegaNext)

    #setting figure and axes of the plot     
    fig, ax = plt.subplots()

    # The initial position of the pendulum rod
    x0, y0 = get_coords(theta0)
    (line,) = ax.plot([0, x0], [0, y0], lw=2, c="k")

    # initial position of bob, with radius of 0.006
    circle = plt.Circle([x0,y0], 0.006, fc="r", zorder=3)
    bob = ax.add_patch(circle)

    # pendulum
    pendulum = [line, bob]

    # Initializing the plot for the animation. The init function is called once at the begining of the animation
    def init():
        ax.set_title("Simple Gravity Pendulum")
        ax.set_xlim(-L * 1.5, L * 1.5)
        ax.set_ylim(-L * 1.5, L * 1.5)
        #returns object that should be updated in each frame
        return pendulum

    # update function is called for each new frame,it modifies the data to be plotted at current frame and returns modified object
    def update(i):
        x, y = get_coords(thetals[i])
        bob.set_center((x, y))
        line.set_data([0, x], [0, y])
        return pendulum

    # Setting up the animation
    # number of frames = total values of theta obtained, and we would loop it over, by keeping repeat = True
    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames= len(thetals),
        repeat=True,
        interval=1,
        blit=True,
    )

    # writer = PillowWriter(fps=10)
    # anim.save("q5.gif",dpi=80,writer=writer);
    # anim.save('q3.gif', fps = 5);
    plt.show()
    return anim

if __name__ == "__main__":
    # solving the ode in the interval t=[0,10]
    # and initial value of t0=0,omega0 = 0, theta0=pi/4, Length of pendulum = 0.1 
   ani = solveODE(t0=0,T=10,g=9.8, L=0.1, omega0=0,theta0=math.pi / 4, h=0.001)