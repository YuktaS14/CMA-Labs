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
    Returns the double derivative of the function sin(x²) at x
"""
def doubleDeriv(x):
    return 2*(math.cos(x**2)-2*(x**2)*math.sin(x**2))

"""
    Returns the triple derivative of the function sin(x²) at x
"""
def tripleDeriv(x):
    return -4*x*(3*math.sin(x**2) + 2*(x**2)*math.cos(x**2))


"""
    Returns the forward finite difference approximation of the function sin(x²) at x
"""
def forwardFinDiff(x,h):
    value = f(x+h)-f(x)
    value = float(value)/float(h)
    return value
"""
    Returns the backward finite difference approximation of the function sin(x²) at x
"""
def backwardFinDiff(x,h):
    value = f(x)-f(x-h)
    value = float(value)/float(h)
    return value

"""
    Returns the centered finite difference approximation of the function sin(x²) at x
"""
def centeredFinDiff(x,h):
    value = f(x+h)-f(x-h)
    value = float(value)/float(2*h)
    return value


# values of h varies from (0,1]. We take 100 points within this interval
points = 100
hls = list(np.linspace(0,1,points))
# h should be a positive finite value, so we remove 0 from possible values of h  
hls.remove(0.0);

"""
    For each value of h we calculate the maximum absolute error of approximations δₕ⁺(x) and δₕᶜ(x) (max of those calculted at each x value)
    And the theoretical maximum absolute error of approximations δₕ⁺(x) and δₕᶜ(x) for the function sin(x²)is obtained by
    calculating double derivative and triple derivative respectively at point ephsilon which lies between x and x+h corresponding to each x value
"""
yApproxffd = []
yTrueffd = []

yApproxcfd = []
yTruecfd = []

for h in hls:
    # initializing the max approx and theoretical values for the errors in ffd and cfd
    maxForwardApprox = 0
    maxCenteredApprox = 0
    theoreticalAbsErrorcfd = 0
    theoreticalAbsErrorffd = 0

    # x lies withing interval(0,1). We consider 100 points within this interval
    x = np.linspace(0,1,100)
    for xi in x:
        #calculating absolute error of approximations
        errorffd = abs(forwardFinDiff(xi,h) - deriv(xi))
        maxForwardApprox = max(maxForwardApprox, errorffd)
        errorcfd = abs(centeredFinDiff(xi,h) - deriv(xi))
        maxCenteredApprox= max(maxCenteredApprox, errorcfd)

        # Now to find theoretical maximum errors we consider ephsilon in the interval [x,x+h]. We consider 100 points in this interval
        ephsilonLs = np.linspace(xi,xi+h,100)
        
        # to get maximum absolute error at particular h and x value, we have to find the maximum double derivative and triple derivative resp. for an ephsilon in the interval [x,x+h]
        double_derivative_max = 0;
        triple_derivative_max = 0;
        for ephsilon in ephsilonLs:
            double_derivative_max = max(double_derivative_max,abs(deriv(ephsilon)))
            triple_derivative_max = max(triple_derivative_max,abs(tripleDeriv(ephsilon)));
        
        theoreticalAbsErrorffd = max(theoreticalAbsErrorffd, double_derivative_max *(float(h)/float(2)))
        theoreticalAbsErrorcfd = max(theoreticalAbsErrorcfd,triple_derivative_max*(float(h**2)/float(6)))

    # Appending the values to the corresponding lists
    yApproxffd.append(maxForwardApprox)
    yApproxcfd.append(maxCenteredApprox)

    yTrueffd.append(theoreticalAbsErrorffd)
    yTruecfd.append(theoreticalAbsErrorcfd)


# plotting the graph for maximum absolute error of approximations and theoretical maximum absolute error of approximations
plt.plot(hls,yApproxffd,'b',label = "Error in forward finite approximation")
plt.plot(hls,yTrueffd,'r',label = "Theoretical error in forward finite approx")
plt.plot(hls,yApproxcfd,'g',label="Error in centered finite approximation")
plt.plot(hls,yTruecfd,'purple',label = "Theoretical error in centered finite approx")

plt.title("Visualizing maximum absolute error of approximations \n δₕ⁺(x) and δₕᶜ(x) for the function sin(x²)")
plt.xlabel("h")
plt.ylabel("Maximum Absolute Error")
plt.legend()
plt.show()



