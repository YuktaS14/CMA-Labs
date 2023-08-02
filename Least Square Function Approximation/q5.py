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

if __name__ == "__main__":
    # Testing the function
    t0 = computeNthChebyshevsPoly(0)
    print(t0)

    t1 = computeNthChebyshevsPoly(1)
    print(t1)

    t2 = computeNthChebyshevsPoly(2)
    print(t2)