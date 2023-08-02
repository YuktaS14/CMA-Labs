import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.integrate as integrate
from matplotlib.animation import FuncAnimation
from q1 import Polynomial

"""
    Function that uses Backward Euler method to solve the ODE x'(t) = -2*x(t)
"""

def backwardEuler(f,getNext,discrSteps,x0,t0,T):
    xls = []
    tls = []
    polys =[]

    # solution for differential equation is found for each discretization step h
    for h in discrSteps:
        xPoints = [x0]
        # generating t points with current step size
        tPoints = list(np.arange(t0,T+h,h))
        
        # computing xpoints using current value of t and x, as per the formula of Backward Euler
        pointsLs = []
        for i in range(len(tPoints)-1):
            xNext = getNext(xPoints[i],h)
            xPoints.append(xNext)
            pointsLs.append((float(tPoints[i]), float(xPoints[i])))
        
        # finding the best fit polynomial for the solution found
        p1 = Polynomial([0])
        p1 = p1.bestFitPolynomial(pointsLs,len(pointsLs)-1)
        
        # storing the solution for each discretiztion step in a list and plotting it later
        xls.append(xPoints)
        tls.append(tPoints)
        polys.append(p1)

        #Printing the polynomial solution for the given ode
        print(f"for h = {h}, Polynomial equation is:")
        print(p1.getPolynomialStr(),"\n")

    # title and axes labels for the plot
    plt.title("Backward Euler Method to solve x'(t) = -2x(t)")
    plt.xlabel("t")
    plt.ylabel("x(t)")
    c = ['b','y',"r","g","m"]

    #plotting the polynomial obtained for each discretization step
    for i in range(len(polys)):
        # plt.plot(tls[i],xls[i],color=c[i])
        x = list(np.linspace(tls[i][0],len(tls[i])-1,100))
        y = list([polys[i][xi] for xi in x])
        plt.plot(x,y,c[i],label = f"h: {discrSteps[i]}")
        plt.plot(tls[i],xls[i],c[i]+'o')

    #computing the actual function value and plotting the actual function
    tpts = list(np.linspace(t0,T,100))
    xpts = list(map(f,tpts))
    plt.plot(tpts,xpts,'k',label="Actual Function")
    plt.ylim(-1,4)
    plt.xlim(0,10)
    plt.legend()
    plt.show() 

# f: x(t) = 5*e(⁻²*ᵗ⁾ 
def f(t):
    return 5*(math.e)**(-2*t)

# xₙ₊₁ = xₙ/(1-(lambda)*h)     : lambda = -2 (from the given ode)
def getNext(x,h):
    return x/(1+2*h)

if __name__ == "__main__":
    # finding the solution for the ode using following discretization steps and initial value of x= 5, t=0, in the interval t = [0,10]
    discrSteps = [0.1,0.5,1,2,3]
    backwardEuler(f,getNext,discrSteps,5,0,10)