import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
import random
from q6 import Polynomial

"""
    Function to compute the zeroes of a continuous function f in interval [a,b] within an error of 10⁻³
"""
def computeZeroes(f,a,b):
    p = Polynomial([])
    n= 15
    epsilon = 0.001
    # Computing the best fit polynomial for the function f using bestFitFunction method of Polynomial class
    p1 = p.bestFitFunction(f,a,b,n)
    # print(p1.getPolynomialStr())

    # Printing the roots of the function within the interval (a,b)
    print(f"Real roots of the function: sin(x) in the interval ({a},{b}) are:")
    
    # roots are found using printRoots method of the Polynomial class
    # since we are considering roots with an interval, we only consider real roots
    roots = p1.printRoots(epsilon,selectReal = 1)

    # printing the roots
    for root in roots:
        if(root>=a and root <= b):
            print(root)

# f(x) = sin(x)
def f(x):
    return np.sin(x)

if __name__ == "__main__":
    #test case
    computeZeroes(f,0,10)