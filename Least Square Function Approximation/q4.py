
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


"""
    Function to compute the least-square approximation of eˣ in the interval [−1, 1] using
the first n Legendre polynomials. 
"""

@handleException
def computeLSA(f,n):
    if not isinstance(n,int):
        raise Exception("n should be an integer.")
    if(n<0):
        raise Exception("n should be a non-negative integer.")
    # list of all aj
    ajs = []
    # list of all legendre polynomials
    legendrePolys = []

    # weight function = 1
    def w(x):
        return 1
    
    # computing n legendre polynomials
    for i in range(n+1):
        lpolynomial = computeNthLegendrePolynomial(i)
        legendrePolys.append(lpolynomial)
    
    # given interval: [-1,1]
    a = -1
    b = 1

    # using the formula to compute the least-squares approximation of f(x) on [-1, 1] as per the lecture
    for j in range(n+1):
        cj= integrate.quad(lambda x: (w(x) * legendrePolys[j][x] * legendrePolys[j][x]),a,b)[0]
        aj = integrate.quad(lambda x: (w(x) * legendrePolys[j][x] * f(x)),a,b)[0]
        ajs.append(aj/cj)
    approxPoly = Polynomial([0])

    for j in range(n+1):
        approxPoly = approxPoly + (ajs[j] * legendrePolys[j])
    
    #consider 50 points in the interval [-1,1]
    x = list(np.linspace(a,b,50))
    y = list(map(f,x))
    # plotting the input points
    plt.plot(x,y,'r',linestyle = 'dashed',dashes=[4,2],linewidth='3',label = "Input function" )
    # plotting the computed polynomial
    plt.title("Least Square Approximation of eˣ :\n{}".format(approxPoly.getPolynomialStr()))
    approxPoly.getplot(a,b)
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.show()
    
    return approxPoly

# function given: eˣ 
def f(x):
    return math.e**x

if __name__ == "__main__":
    p1 = computeLSA(f, 3)
    print(p1)