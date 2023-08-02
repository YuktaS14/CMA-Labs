import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import math
from q1 import Polynomial

"""
    Using function decorators to handle exceptions.
"""
def handleException(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args,**kwargs)
        except Exception as e:
            print(type(e))
            print(e)
            return

            # Uncomment following line if program should exit after an exception occurs
            # sys.exit(1);
    return wrapper

"""
    Function to compute nᵗʰ Chebyshevs Polynomial
"""
@handleException
def computeNthChebyshevsPoly(n):
    if not isinstance(n,int):
        raise Exception("n should be an integer.")
    if(n<0):
        raise Exception("n should be a non-negative integer.")
    
    # T₀(x) = 1, T₁(x) = x
    if(n == 0):
        return Polynomial([1])
    if(n == 1):
        return Polynomial([0,1])

    # computing nᵗʰ Chebyshev's polynomial using recurrence relation:  Tn+1(x) = 2xTn(x) − Tn−1(x)    
    t0 = Polynomial([1])
    t1 = Polynomial([0,1])
    tnth = 2*Polynomial([0,1])* computeNthChebyshevsPoly(n-1) - computeNthChebyshevsPoly(n-2)
    return tnth


"""
    Function to demonstrate that first 5 Chebyshev polynomials are numerically orthogonal with respect the weight function w(x) = 1/√(1-x²) in the interval [-1, 1] 
"""
@handleException
def verifyOrthogonality():
    a = -1
    b = 1

    # weight function:   w(x) = 1/√(1-x²)
    def w(x):
        root = math.sqrt(1-x**2)
        return 1/root
    
    # computing first 5 Chebyshev polynomials
    cPolys = []
    p = Polynomial([])
    for i in range(5):
        c = computeNthChebyshevsPoly(i)
        cPolys.append(c)
    
    # creating a matrix to store the integral (w(x)* φi (x)* φj(x)) dx  in the interval [a,b] for each i,j belonging to the cPolys
    # according to the orthogonal functions property this matrix should be a diagonal matrix as for all i != j, integral evaluates to 0
    mat = []
    for i in range(5):
        r = []
        for j in range(5):
            p = integrate.quad(lambda x : w(x)*cPolys[i][x]*cPolys[j][x],a,b)[0]
            r.append(round(p,2))
        mat.append(r)
    
    print(mat)


if __name__ == "__main__":
    # Testing the function
    verifyOrthogonality()