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


# x interval is between [0,1]. We plot graph for 50 points within this interval
x = np.linspace(0,1,50)

# plotting the graph for:
# 1.absolute error of approximation for forward finite difference approximation for the input function in the interval [0,1]
ffd = list([abs(forwardFinDiff(xi)-deriv(xi)) for xi in x])

# 2.absolute error of approximation for backward finite difference approximation for the input function in the interval [0,1]
bfd = list([abs(backwardFinDiff(xi)-deriv(xi)) for xi in x])

# 3.absolute error of approximation for centered finite difference approximation for the input function in the interval [0,1]
cfd = list([abs(centeredFinDiff(xi)-deriv(xi)) for xi in x])

plt.title('Visualization of absolute errors of approximation \n(δ⁻), (δ⁺) and (δᶜ) of the function sin(x²) ')
plt.xlabel('x')
plt.ylabel('Absolute Error')

plt.plot(x,ffd,'r',label="forward finite diff")
plt.plot(x,bfd,'b', label="backward finite diff")
plt.plot(x,cfd,'k--', label="centered finite diff")
plt.legend()
plt.show()