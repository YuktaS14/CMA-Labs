import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
import random
from scipy import linalg, integrate


"""
    Function to find the solution x = [x1,x2,x3] to the given functions using Newton-Raphson method 
"""
def computeFxk(x0,k):
    # computing the function value f(x) = [f1(x1), f2(x2), f3(x3)]
    def f(x):
        x1,x2,x3 = x
        f1 = 3*x1 - math.cos(x2*x3) - 3/2
        f2 = 4*(x1**2) -625*(x2**2) + 2*x3 -1
        f3 = 20*x3 + math.exp(-1*x1*x2)+9
        return [f1,f2,f3]
    
    # Computing the Jacobian Matrix for given functions
    def fJacobi(x):
        x1,x2,x3 = x
        j1 = [3, x3* math.sin(x2*x3), x2*math.sin(x2*x3)]
        j2 = [8*x1,-1250*x2,2]
        j3 = [-x2*math.exp(-1*x1*x2), -x1*math.exp(-1*x1*x2),20]
        return [j1,j2,j3]
    
    
    #xlsNorm consists of list of ||f(xₖ)|| for each iteration
    xlsNorm = [scipy.linalg.norm(f(x0))] 
    x = x0
    # k is the number of iterations to find final root
    numIterations = k
    # root is found iteratively using Newton Raphson formula
    while(k>0):
        xNext = x- (scipy.linalg.inv(fJacobi(x)))@f(x)
        # append the norm (||f(xₖ)||) for each x found during iterations
        xlsNorm.append(scipy.linalg.norm(f(xNext)))
        #updating the current x value
        x = xNext
        #decrementing the iteration counter
        k-=1

    # x calculated after kth iteration will be the final root 
    print(f"Root of the given function is: {x}")

    # plotting the graph between ||f(xₖ)|| vs k
    tls = np.linspace(1,numIterations+1,numIterations+1)
    plt.plot(tls,xlsNorm,'b',label="||f(xₖ)||")
    plt.xlabel("Number of Iterations (k)")
    plt.ylabel("||f(xₖ)||")
    plt.title("||f(xₖ)|| vs k")
    plt.grid()
    plt.legend()
    plt.show()

    
if __name__ == "__main__":
    #test case
    computeFxk([1,2,3],30)