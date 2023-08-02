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
    function to compute coefficients of the best-fit Fourier approximation Sn(x) of function f(x)
"""
@handleException
def bestFitFourier(f,n):
    # handling exception 
    if not isinstance(n,int):
        raise Exception("n should be an integer.")
    if(n<0):
        raise Exception("n should be a non-negative integer.")
    
    # given interval: [-pi,pi] 
    a = -1*math.pi
    b = math.pi
    
    # computing ak and bk for k = 0 to n
    ak = []
    bk = []
    for k in range(0,n+1):
        aik =  (1/math.pi) * integrate.quad(lambda x: f(x)*math.cos(k*x),a,b)[0]
        bik =  (1/math.pi) * integrate.quad(lambda x: f(x)*math.sin(k*x),a,b)[0]
        ak.append(aik)
        bk.append(bik)
    
    #printing the  coefficients ak and bk
    print("coefficients of S{} (x): \n".format(n))
    for i in range(n+1):
        print("a{} : {} b{} : {}".format(i,ak[i],i,bk[i]))
    
    # computing the value of function for different values of x
    x = list(np.linspace(a,b,100))
    y = list(map(f,x))

    # computing the value of Sₙ(x) for different x points
    y2 = []
    for xi in x:
        aksum = 0
        bksum = 0
        for k in range(1,n+1):
            aksum += ak[k]* math.cos(k*xi)
            bksum += bk[k]* math.sin(k*xi)
        sn = ak[0]/2 + aksum + bksum
        y2.append(sn)

     # plotting the input function as well as the best fit fourier series for the given function
    plt.title("Best Fit Fourier Series")
    plt.plot(x,y,'b',label = "Input function" )
    plt.plot(x,y2,'r',label='Fourier approximation: Sn(x)')
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.show()

# function given: eˣ 
def f(x):
    return math.e**x

if __name__ == "__main__":
    # Testing the function
    bestFitFourier(f, 10)
