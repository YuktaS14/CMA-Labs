import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
import random

def computeNthRoot(n,a,ephsilon):
    
    # Defining the equation to solve: we have to find Nth root of a
    def NthRootEq(x):
        return x**n - a
    
    # Setting initial values for low and high bounds of the search range
    low = 0; high = a

    # Using bisection method to find the root of the equation
    while (abs(high-low) > ephsilon):
        mid = (low+high)/2
        
        # If the midpoint satisfies the equation, we have found the root so we return it
        if(NthRootEq(mid) == 0):
            return mid
        
        # If the product of the function values at midpoint and high is positive, then it means that both are at the same side of x axis,
        # so we set the new high to midpoint
        elif(NthRootEq(mid)* NthRootEq(high) > 0):
            high = mid
        
        # else if they are to opposite sides of x axis, then the product of the function values at midpoint and high is negative,
        # then we set the new low to midpoint, to get closer to root
        else:
            low = mid
    
    # Return the midpoint of the final range as the root approximation
    return (high+low)/2
    

if __name__ == "__main__":
    # test case
    n = 15
    a = 5**15
    eps = 0.00001
    print(f"The {n}th root of {a} is {computeNthRoot(n, a, eps)}")