import matplotlib.pyplot as plt
import numpy as np
import random
import math
from scipy import misc
import scipy.integrate as integrate

"""
    Returns the value of function 2.x.e^(x²) at x
"""

def func(x):
    return 2*x*(math.e **(x**2))


"""
    Returns the value of integral for the function 2.x.e^(x²) at x.
    Integral is: e^(x²)
"""
def integralFunc(x):
    return math.e**(x**2)
    
# u varies from 0 to 3. We consider 100 points within this interval for u
umin = 0
umax = 3
numOfPoints = 100
uLs = list(np.linspace(umin,umax,numOfPoints))
uLs.remove(0)

# Area under the graph as computed using different integration functions are plotted on a graph for corresponding interval [0,u]
actualArea = []
quadIntegral = []
trapezoidIntegral = []
simpsonIntegral = []
rombIntegral = []


for u in uLs:
    # for each u, x points are chosen from the interval [0,u]
    # corresponding function values at each x is found and appended in y
    # x and y values are taken as input by trapezoid and simpson integral functions
    x = np.linspace(umin,u,50)
    y = list(map(func,x))

    # Actual area in each interval [0,u] is found
    actualArea.append(integralFunc(u)-integralFunc(umin))
    # General purpose integation
    quadIntegral.append(integrate.quad(func,umin,u)[0])
    # Uses trapezoidal rule to compute integral 
    trapezoidIntegral.append(integrate.trapezoid(y,x))
    # Simpson Integration
    simpsonIntegral.append(integrate.simpson(y,x))
    # Romberg Integration
    rombIntegral.append(integrate.romberg(func,umin,u))


# Plotting the graph with respect to areas given by different integral functions
plt.plot(uLs,actualArea,'r-', label = "Actual Area",linewidth = "4")
plt.plot(uLs,quadIntegral,'b--',label = "Quad Integral", linewidth = "4")
plt.plot(uLs,trapezoidIntegral,'g',label = "Trapezoid integral",linewidth = "2")
plt.plot(uLs,simpsonIntegral,'y' , label = "Simpson integral",linewidth = "3")
plt.plot(uLs,rombIntegral,'purple',label = "Romberg integral")

plt.title("Visualizing various integration functions in \n Python's scipy.integrate module for the function 2.x.e^(x²) ")
    
plt.xlabel("u")
plt.ylabel("Area under the graph")
plt.legend()
plt.show()

