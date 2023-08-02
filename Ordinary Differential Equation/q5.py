import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.integrate as integrate
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation, PillowWriter

"""
    Function to find solution to the Three Body Problem given the ODEs
"""
def solveODE(r10,r20,r30,v10,v20,v30,t0,T):
    # considering step size = 0.2
    h = 0.2
    # generating points with h as step size
    tls = np.arange(t0,T+h,h)

    #Returns the norm of the vector (r2 - r1). If norm is 0, returns some dummy value, just to avoid the value (val) from going to inf
    def getnorm(r1, r2):
        return max(np.linalg.norm(r2 - r1),10)

    # computing d²(r)/d(t²) for all the 3 bodies, acc to ODE given
    def ddrdt(r1,r2,r3):
        r1 = np.array(r1)
        r2 = np.array(r2)
        r3 = np.array(r3)
        val = ((r2-r1)/(getnorm(r2,r1))**3) + ((r3-r1)/(getnorm(r3,r1))**3)
        return list(val)

    # We have 6 one ordered ODEs => dv/dt and dr/dt corresponding to each of the 3 bodies   
    def derivatives(t,y):
        # each of the r, v is a vector of 2 coordinates(x,y)
        # so in total y has 12 parameters 
        r1x,r1y,r2x,r2y,r3x,r3y,v1x,v1y,v2x,v2y,v3x,v3y = y
        r1 = [r1x, r1y]
        r2 = [r2x, r2y]
        r3 = [r3x, r3y]

        v1 = [v1x,v1y]
        v2 = [v2x,v2y]
        v3 = [v3x,v3y]

        # calculating dv/dt
        dv1 = ddrdt(r1,r2,r3)
        dv2 = ddrdt(r2,r3,r1)
        dv3 = ddrdt(r3,r1,r2)

        # returns an array with 12 parameters
        ret = np.array([v1,v2,v3,dv1,dv2,dv3])
        return ret.flatten()
    
    # initial values
    y0 = np.array([r10,r20,r30,v10,v20,v30])

    # solving the ODEs using solve_ivp function in the interval [t0,T]
    soln = solve_ivp(fun=derivatives,t_span=[t0,T],y0=y0.flatten(),t_eval=tls)

    # getting all the values of r and v for each body in the interval [t0,T] from the solution obtained
    r1x,r1y,r2x,r2y,r3x,r3y,v1x,v1y,v2x,v2y,v3x,v3y = soln.y
    
    # setting figure and axes of the plot
    fig,ax = plt.subplots()
    # radius of balls(bodies) = 0.1
    radius = 0.1
    # setting the initial positions of the 3 bodies
    b1 = ax.add_patch(plt.Circle((r10[0],r10[1]),radius,fc="r",label = "Body1" ))
    b2 = ax.add_patch(plt.Circle((r20[0],r20[1]),radius,fc="b",label = "Body2" ))
    b3 = ax.add_patch(plt.Circle((r30[0],r30[1]),radius,fc="g",label = "Body3" ))

    # object consisting of 3 bodies
    patches = [b1,b2,b3]

    ax.set_title("Three-Body Problem")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    # Initializing the plot for the animation. The init function is called once at the begining of the animation
    def init():
        ax.set_title("Three-Body Problem")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        # Set the plot limits
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)

        #returns object that should be updated in each frame
        return patches
    
    # update function is called for each new frame,it modifies the data to be plotted at current frame and returns modified object
    def update(i):
        b1.set_center((r1x[i],r1y[i]))
        b2.set_center((r2x[i],r2y[i]))
        b3.set_center((r3x[i],r3y[i]))
        return patches
    
    # Setting up the animation
    anim = FuncAnimation(fig,update,init_func=init,frames=len(tls),interval = 1,repeat=True,blit=True)

    # writer = PillowWriter(fps=10)
    # anim.save("q5.gif",dpi=80,writer=writer);
    plt.legend()
    plt.show()

if __name__ == "__main__":
    r10 = [0,-2.31]
    r20 = [1.732,3]
    r30 = [-1.732,3]
    v10 = [0,0]
    v20 = [0,0]
    v30 = [0,0]

    # solving the ODE in the interval t = [0,400] with initial values of r and v as specified above
    ani = solveODE(r10,r20,r30,v10,v20,v30,t0=0,T=400)