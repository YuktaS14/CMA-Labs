
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
    Function to compute the nᵗʰ Legendre polynomial
"""
@handleException
def computeNthLegendrePolynomial(n):
    if not isinstance(n,int):
        raise Exception("n should be an integer.")
    if(n<0):
        raise Exception("n should be a non-negative integer.")
    
    # x²-1
    p = Polynomial([-1,0,1])
    #computing the numerator for nᵗʰ Legendre polynomial
    numerator = Polynomial([1])
    for i in range(n):
        numerator = p*numerator    
    
    # computing the nᵗʰ derivative of the numerator
    for i in range(n):
        numerator = numerator.derivative()
    # computing the denominator: 2ⁿ * n!
    denominator = 2**n
    for i in range(1,n+1):
        denominator = denominator*i
    #computing the nᵗʰ Legendre polynomial
    poly = (1/denominator)*(numerator)
    return poly


if __name__ == "__main__":
# First Legendre Polynomial
    l0 = computeNthLegendrePolynomial(0)
    print(l0)

    # Second Legendre Polynomial
    l1 = computeNthLegendrePolynomial(1)
    print(l1)

    # Third Legendre Polynomial
    l2 = computeNthLegendrePolynomial(2)
    print(l2)

    # Fourth Legendre Polynomial
    l3 = computeNthLegendrePolynomial(3)
    print(l3)