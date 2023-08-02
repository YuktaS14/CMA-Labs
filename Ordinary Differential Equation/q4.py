import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.integrate as integrate
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks


"""
    Function to find solution to the ODE of Van der Pol Equation
"""
def solveODE(mu,x0,z0,t0,T,h):
    tls = np.arange(t0,T+h,h)

    # This function takes two arguments: t, the current time, and y, a NumPy array containing the current values of the variables(x,z)
    def getDeriv(t,y):
        #computing z=dx/dt and dz/dt = mu*(1-x²)*z - x
        derivs = np.zeros(2)
        x = y[0]; z = y[1]
        derivs[0] = z
        derivs[1] = mu*(1-x**2)*z - x
        return derivs

    # The first row of sol.y contains the values of the first variable of the
    #  Van der Pol equation(->x), and the second row contains the values of the second variable(->z).
    soln = solve_ivp(fun = getDeriv, t_span = [t0,T],y0=[x0,z0],t_eval=tls)
    
    # Find the indices of the peaks of the solution
    peaks, _ = find_peaks(soln.y[0])

    # Compute the period of the limit cycle, which is the time elapsed between two consecutive peaks.
    # taking mean of all to get average time period
    period = np.mean(np.diff(soln.t[peaks]))
    print(f"Time period of the limit cycle for mu = {mu} is: {period:.4f}")

    # plotting the solution for the ode describing Van der Pol equation
    plt.title(f"Van der Pol equation for μ = {mu}")
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.plot(soln.t,soln.y[0])
    plt.grid()
    plt.show()

                   


if __name__ == "__main__":
    # solving the ode for Van der Pol equation in the interval t0=[0,200] with initial value of x0 = 0,z0 = 10 and mu = 2
    solveODE(mu=2,x0=0, z0=10, t0=0, T=200, h=0.001)