import matplotlib.pyplot as plt
import numpy as np
import random
import math


"""
    Returns the value of function 2.x.e^(x²) at x
"""
def fun(x):
    return 2*x*(math.e ** (x**2))

"""
    Returns the value of integral for the function 2.x.e^(x²) at x.
    Integral is: e^(x²)
"""
def integralFun(x):
    return math.e**(x**2)


# Given interval for x is [1,3]
a = 1
b = 3
# area under the curve for interval [1,x] is appended into y
y = []


for M in range(1,51):
    # for each value of M 
    # (the height of each interval)/2 is calculated by following formula
    area = float(b-a)/float(2*M)
    
    # We take M+1 points in the interval [1,3].. (simce there are M intervals )
    x = list(np.linspace(1,3,M+1))

    # then we sum over all the parallel sides of trapezium (for each interval)
    sumOverIntervals = 0
    for k in range(1,M+1):
        sumOverIntervals += (fun(x[k])+fun(x[k-1]))
    #and multiply with height of each interval to get the area under the curve in the interval [1,3]
    area *= sumOverIntervals
    y.append(abs(area))

# plotting the graph between M and the area under the curve corresponding to M
M = list([x for x in range(1,51)])
plt.plot(M,y,'r',label="area using trapezoidal formula", linewidth="3")

# Plotting the exact area under the curve of function 2.x.e^(x²) in the interval [1,3]
exactArea = integralFun(b)-integralFun(a)
plt.axhline(y = exactArea,c='b',label="Exact area")

plt.title("Visualizing area under curve \nas a function of M(number of intervals)")
plt.xlabel("M")
plt.ylabel("area")
plt.legend()
plt.plot()
plt.show()
    