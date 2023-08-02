import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from q1 import Polynomial
import math

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
    Function to compute the polynomil of degree n that best approximates a function f
"""
@handleException
def bestFitFunction(p,f,a,b,n):
    # handling exception
    if not isinstance(n,int):
        raise Exception("Degree of polynomial should be an integer.")
    if(n<0):
        raise Exception("Degree of polynomial should be a non-negative integer.")
    if(a>b):
        raise Exception("In interval a,b: a should be less than b")
    
    # Using formula for the least square approximation of a function
    B = []
    for j in range(n+1):
        rowEntry = integrate.quad(lambda x: x**(j) * f(x),a,b)[0]
        B.append(rowEntry)
    
    A = []
    for j in range(n+1):
        row = []
        for k in range(n+1):
            rowEntry = integrate.quad(lambda x: x**(j+k),a,b)[0]
            row.append(rowEntry)
        A.append(row)
    
    # solving the linear equations
    c = list(np.linalg.solve(A,B));
    p1 = Polynomial(c)

    # plotting the input points
    x = list(np.linspace(a,b,100))
    y = list(map(f,x))
    plt.plot(x,y,'r',linestyle = 'dashed',dashes=[4,2],linewidth='3',label = "Input function" )

    # plotting the computed polynomial
    plt.title("Best Fit Polynomial: \n{}".format(p1.getPolynomialStr()))
    p1.getplot(a,b)
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.show()
    return p1

def func(x):
    return math.sin(x) + math.cos(x)

if __name__ == "__main__":
    p = Polynomial([])
    p1 = bestFitFunction (p,func, float(0), float(math.pi), 5)
    print(p1)