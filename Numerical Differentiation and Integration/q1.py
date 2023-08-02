import matplotlib.pyplot as plt
import numpy as np
import random
import math

"""
    Returns the value of function sin(x²) at x
"""
def f(x):
    return math.sin(x**2)
"""
    Returns the derivativeof function sin(x²) at x >> derivative : 2 * x * cos(x²)
"""
def deriv(x):
    return math.cos(x**2)*2*x

"""
    returns the forward finite difference approximation of the function sin(x²) at x, with h value as 0.01
"""
def forwardFinDiff(x):
    h = 0.01
    value = f(x+h)-f(x)
    value = float(value/h)
    return value

"""
    returns the backward finite difference approximation of the function sin(x²) at x, with h value as 0.01
"""
def backwardFinDiff(x):
    h = 0.01
    value = f(x)-f(x-h)
    value = float(value/h)
    return value

"""
    returns the centered finite difference approximation of the function sin(x²) at x, with h value as 0.01
"""
def centeredFinDiff(x):
    h = 0.01
    value = f(x+h)-f(x-h)
    value = float(value/(2*h))
    return value

# x interval is between [0,1]. We plot graph for 40 points within this interval
x = np.linspace(0,1,40)

# plotting the graph for the actual derivative (f'(x)) and forward finite difference approximation of the input function in the interval [0,1]
y1 = list([deriv(xi) for xi in x])
y2 = list([forwardFinDiff(xi) for xi in x])

plt.title('Visualization of actual derivative (f\'(x)) and \n forward finite difference approximation of the function sin(x²) ')
plt.xlabel('x')
plt.ylabel('f\'(x) and δ⁺')
plt.plot(x,y1,'bo', label = "Actual Derivative")
plt.plot(x,y2,'r', label = "Forward Finite Difference Approximation")
plt.legend()
plt.show()
