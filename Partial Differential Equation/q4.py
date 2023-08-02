import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
import random



"""
    function to compare the convergence rates of the Newton Raphson and Secant methods
"""
def compareConvergence(x0, x1, numIterations):
    # Calculating f(x) = x*eˣ 
    def f(x):
        return x * np.exp(x)
    
    # Calculating f'(x) = (x+1)*eˣ 
    def fDeriv(x):
        return (x+1) * np.exp(x)
    
    # Definining the Newton Raphson method for finding roots
    def newtonRaphsonMethod(x0, numIterations):
        # list to store the differences between consecutive x values
        xls = []  
        # Setting the initial guess for the root
        x = x0  
        # k is the number of iterations to perform
        k = numIterations  
        
        # Looping over the specified number of iterations. Calculating next approximation of the root.
        while(k > 0):
            # Calculating the next approximation of the root, according to newton raphson method
            xNext = x - f(x) / fDeriv(x) 
            # Add the difference between the current and next x values to the list
            xls.append(xNext - x) 
            # Set the current x value to the next x value
            x = xNext 
            # Decrement the iteration counter
            k -= 1  


        # Return the list of differences between consecutive x values
        return xls  
    
    # Define the Secant method for finding roots
    def secantMethod(x0, x1, numIterations):
         # Creating empty list to store the differences between consecutive x values
        tls = [] 
        # Setting the first initial guess for the root
        tprev = x0  
        # Setting the second initial guess for the root
        t = x1         
        # Setting the number of iterations to perform
        k = numIterations  
        
        # Loop over the specified number of iterations
        while(k > 0):
            # Calculating the next approximation of the root according to Secant method
            tNext = t - f(t) * ((t - tprev) / (f(t) - f(tprev)))  
            # Adding the difference between the current and next root approximations to the list
            tls.append(tNext - t)  
            # incrementing tprev and t
            tprev = t 
            t = tNext  
            k -= 1  

        # Return the list of differences between consecutive root values
        return tls  
    
    # Define a function to plot the convergence rates of the two methods
    def plotConvergence():
        # Calculate the differences between consecutive x values for the Newton Raphson method
        xls = newtonRaphsonMethod(x0, numIterations) 
        # Calculate the differences between consecutive x values for the Secant method
        tls = secantMethod(x0, x1, numIterations)  

        # Plotting the difference between consecutive roots found in the iteration vs the iteration number for both the methods
        pts = list(np.linspace(1, numIterations, numIterations)) 
        plt.plot(pts, xls, 'r', label='Newton Raphson Method')  
        plt.plot(pts, tls, 'b', label='Secant Method') 
        # Set the title of the plot
        plt.title("Comparing rate of convegence of\n Newton Raphson and Secant Method") 
        # Set the label for the x-axis
        plt.xlabel('Number of Iterations') 
        # Set the label for the y-axis
        plt.ylabel("Difference between 2 consecutive x values") 
        plt.ylim(-4,1)
        plt.legend()
        plt.show()

    plotConvergence()


if __name__ == "__main__":
    # Comapring the convergence from the plot
    compareConvergence(200,210,300)