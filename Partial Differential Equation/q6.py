
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
import random
from scipy import linalg, integrate




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


class Polynomial():
    @handleException
    def __init__(self,coeff):
        if not isinstance(coeff,list):
            raise Exception("Coefficients of polynomial should be passed as a list")
    
        self.coeff = coeff;
    
    @handleException
    def __str__(self):
        # returns coefficient of polynomial as a string
        s = "Coefficients of the polynomial are:\n"
        for i in self.coeff:
            s+= str(i)+" ";
        return s;

    @handleException
    def __add__(self,p2):
        if not isinstance(p2,Polynomial):
            raise Exception ("Invalid Input of polynomial")
        
        #finding the degrees of each polynomial
        d1 = len(self.coeff)
        d2 = len(p2.coeff)
        maxi  = max(d1,d2);

        # appending the addition of 2 coefficients of same degree, if they exist.
        # if not, appending the cefficient + 0 for that resp. degree
        p3 = []
        for i in range (maxi):
            if i<d1 and i<d2 :
                p3.append((self.coeff[i] + p2.coeff[i]));
            elif i<d1:
                p3.append(self.coeff[i])
            elif i<d2:
                p3.append(p2.coeff[i])

        #returns the resultant polynomial
        return Polynomial(p3);

    @handleException
    def __sub__(self,p2):
        if not isinstance(p2,Polynomial):
            raise Exception ("Invalid Input of polynomial")
        
        #finding the degrees of each polynomial
        d1 = len(self.coeff)
        d2 = len(p2.coeff)
        maxi = max(d1,d2)

        p3 = []
        for i in range(maxi):
            # for degree i, if coefficients exist in both polynomials then subtract those and append to resulting polynomial
            if i<d1 and i<d2:
                p3.append(self.coeff[i] - p2.coeff[i])
            # if not, correspondingly append the coefficient for the degree i depending on whether it is from polynomial self or p2 
            elif i<d1:
                p3.append(self.coeff[i])
            elif i<d2:
                p3.append(-p2.coeff[i])

        #returns the resultant polynomial
        return Polynomial(p3)

    @handleException
    def __getitem__(self,x):
        # evaluates the polynomial at given x and returns the value val
        deg = len(self.coeff)-1
        val = 0;
        for i in range (deg+1):
            val += self.coeff[i]*(x**i)
        return val;
        
    @handleException
    def __rmul__(self,scalar):
        if not isinstance(scalar, (int, float)):
            raise Exception("Invalid scalar type")
        # returns the polynomial whose each coefficient is multiplied by the scalar
        p2 = [i*scalar for i in self.coeff]
        return Polynomial(p2);

    @handleException
    def __mul__(self,p2):
        #overloading * operator to allow multiplication of polynomials
        if not isinstance(p2, Polynomial):
            raise Exception("Invalid Polynomial input")

        #degrees of given polynomials
        p1Deg = len(self.coeff)-1;
        p2Deg = len(p2.coeff)-1;

        # creating dictionaries to store the coefficients corresponding to each degree for the given polynomials
        d1 = {};
        d2 = {};

        for i,v in enumerate(self.coeff):
            d1[p1Deg-i] = v;
        for i,v in enumerate(p2.coeff):
            d2[p2Deg-i] = v;

        #resultant polynomial will have at max degree = sum of the degrees of the polynomials multiplied
        p3Deg = p1Deg+p2Deg;

        # creating coefficients list for the resultant polynomial
        p3 = [0]*(p3Deg+1);

     
        for i in d2.keys():
            for j in d1.keys():
                # updating the coefficient of degree(p3Deg-(i+j)) with the corresponding value 
                p3[p3Deg - (i+j)] += d2[i]*d1[j];
        
        return Polynomial(p3)

    @handleException
    def getPolynomialStr(self):
        # function to get the polynomial string in correct format
        def getSuperscript(n):
            # function to get superscript reprentation of numbers
            SSDigits = str.maketrans("-0123456789","⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
            return str(n).translate(SSDigits)

        # getting title for the plot - a polynomial in correct format    
        deg = len(self.coeff)-1;
        s = ""
        for i in range(deg+1):
            if self.coeff[i] == 0:
                continue;
                
            if i == 0:
                s+= str(round(self.coeff[i],4))
                continue;

            if(self.coeff[i]>0):
                s+="+"
            if(self.coeff[i]<0):
                s+="-"
            if(abs(self.coeff[i]) != 1):
                s+= str(round(abs(self.coeff[i]),4))
            
            if i == 1:
                s+= "x"
            else:
                s += "x"+ getSuperscript(i)

        return s


    @handleException
    def getplot(self,a,b):
        # function to plot the polynomial within the x interval (a,b)
        xmin = min(a,b)
        xmax = max(a,b);

        #plotting for 100 points
        x = list(np.linspace(xmin,xmax,100))
        y = list([self[xi] for xi in x])
        plt.plot(x,y,'b', label = "polynomial")

    @handleException
    def show(self,a,b):
        if not isinstance(a,(float,int)) or not isinstance(b,(float,int)):
            raise Exception("range of x should be either of type float or int")

        # plotting the polynomial
        self.getplot(a,b)
        plt.grid()
        plt.xlabel('x')
        plt.ylabel('P(x)')
        plt.title("Plot of the polynomial {}".format(self.getPolynomialStr()))        
        plt.show()
     
    @handleException
    def fitViaMatrixMethod(self,ls):
        # exception handling
        if not isinstance(ls,list):
            raise Exception("Invalid Input. List of points(x,y) is expected")
        for i in ls:
            if not isinstance(i,tuple) or len(i) != 2:
                raise Exception("Points should be specified in form of tuple (x,y)")
            if not isinstance(i[0],(float,int)) or not isinstance(i[1],(float,int)):
                raise Exception("x,y should be of type float or int");
        
        # calculating degree of the polynomial
        deg = len(ls)-1;

        
        A = []
        b = []

        # from given points creating the matrix A and the vector b resp.
        for p in ls:
            r = [p[0]**i for i in range(deg+1)];
            A.append(r);
            b.append([p[1]]);

        # for plotting the polynomial we need the x and y coordinates of the points
        x = list([p[0] for p in ls])
        y = list([p[1] for p in ls])

        # using linear system, we try to compute the polynomial which passes through the given points
        c = np.linalg.solve(A,b);
        
        t = []
        for i in c:
            t.append(round(i[0],2))
        # print(t)
        p1 = Polynomial(t);

        # plotting the computed polynomial
        plt.plot(x,y,'ro')

        # sorting values of x to get interval in which we need to plot the graph -> (a,b)
        x.sort()

        p1.getplot(x[0],x[(len(x)-1)])
        plt.grid()
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Polynomial interpolation using matrix method gives polynomial: {}".format(p1.getPolynomialStr()))
        plt.show()

    @handleException
    def fitViaLagrangePoly(self,ls):
        # handling exceptions
        if not isinstance(ls,list):
            raise Exception("Invalid Input. List of points(x,y) is expected")
        for i in ls:
            if not isinstance(i,tuple) or len(i) != 2:
                raise Exception("Points should be specified in form of tuple (x,y)")
            if not isinstance(i[0],(float,int)) or not isinstance(i[1],(float,int)):
                raise Exception("(x,y) should be of type float or int");

        # calculating degree of the polynomial
        deg = len(ls)-1;
        A = []
        b = []

        # from given points creating the matrix A and the vector b resp.
        for p in ls:
            r = [p[0]**i for i in range(deg+1)];
            A.append(r);
            b.append([p[1]]);

        # for plotting the polynomial we need the x and y coordinates of the points
        x = list([p[0] for p in ls])
        y = list([p[1] for p in ls])

        # initializing the resultant polynomial which would be linear combination of lagrange's polynomials (phi j) 
        resultant = Polynomial([0]);
        for i in range(deg+1):
            #computing lagrange's polynomial corresponding to each point
            numerator = Polynomial([1])
            denominator = 1

            for p in range(0,len(ls)):
                if(p == i):
                    continue;
                numerator = numerator * Polynomial([-x[p],1])
                denominator = denominator*(x[i]-x[p]);
            LPolynomial = [float(x/denominator) for x in numerator.coeff]
            LPolynomial = Polynomial(LPolynomial)
            resultant = resultant + y[i]*LPolynomial


        
        # plotting the computed polynomial
        plt.plot(x,y,'ro')

        # sorting values of x to get interval in which we need to plot the graph -> (a,b)
        x.sort()

        resultant.getplot(x[0],x[(len(x)-1)])
        plt.grid()
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Interpolation using Lagrange polynomial gives polynomial: {}".format(resultant.getPolynomialStr()))
        plt.show()

    @handleException
    def derivative(self):
        dv = []
        for i in range(1,len(self.coeff)):
            d = i*self.coeff[i]
            dv.append(d)
        return Polynomial(dv)

    @handleException
    def integral(self):
        self.integralPoly = [0]
        for i in range(len(self.coeff)):
            v = self.coeff[i]/(i+1)
            self.integralPoly.append(v)
    
    @handleException
    def integralValue(self,x):
        ans = 0;
        for i in range(len(self.integralPoly)):
            ans += self.integralPoly[i]* (x**i)
        return ans
    
    @handleException
    def area(self, a, b):
        self.integral()
        return (self.integralValue(b) - self.integralValue(a))    
       
    """
        function to compute the polynomial of degree n, that is the best fit for given set of points
    """
    @handleException
    def bestFitPolynomial(self,ls, n):
        # handling exceptions
        if not isinstance(ls,list):
            raise Exception("Invalid Input. List of points(x,y) is expected")
        for i in ls:
            if not isinstance(i,tuple) or len(i) != 2:
                raise Exception("Points should be specified in form of tuple (x,y)")
            if not isinstance(i[0],(float,int)) or not isinstance(i[1],(float,int)):
                raise Exception("(x,y) should be of type float or int");
        

        if not isinstance(n,int):
            raise Exception("Degree of polynomial should be an integer.")
        if(n<0):
            raise Exception("Degree of polynomial should be a non-negative integer.")

        # m = total number of points
        m = len(ls)

        # separating x and y coordinates
        x = [pt[0] for pt in ls]
        y = [pt[1] for pt in ls]

        # computing vector b
        b = []

        # jᵗʰ row of b = summation (yi*xiʲ) where i varies from 0 to m
        for j in range(0,n+1):
            rowEntry = 0
            for i in range(1,m+1):
                rowEntry += y[i-1]* (x[i-1] ** j)
            b.append(rowEntry)

        # computing matrix A
        A = []

        # using the formula mentioned in lecture
        for j in range(0,n+1):
            a = []
            for k in range(0,n+1):
                rowEntry = 0
                for i in range(1,m+1):
                    rowEntry += x[i-1]**(j+k)
                a.append(rowEntry)
            A.append (a)

        
        # using linear system, we try to compute the polynomial which passes through the given points
        c = list(np.linalg.solve(A,b));
  
        p1 = Polynomial(c)

        # plotting the input points
        plt.plot(x,y,'ro',label = "input points" )

        # sorting values of x to get interval in which we need to plot the graph -> (a,b)
        x.sort()

        # plotting the computed polynomial
        plt.title("Best Fit Polynomial: \n{}".format(p1.getPolynomialStr()))
        p1.getplot(x[0],x[(len(x)-1)])
        plt.grid()
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.show()
        return p1
    @handleException
    def bestFitFunction(self,f,a,b,n):
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
            rowEntry = scipy.integrate.quad(lambda x: x**(j) * f(x),a,b)[0]
            B.append(rowEntry)
        
        A = []
        for j in range(n+1):
            row = []
            for k in range(n+1):
                rowEntry = scipy.integrate.quad(lambda x: x**(j+k),a,b)[0]
                row.append(rowEntry)
            A.append(row)
        
        # solving the linear equations
        c = list(np.linalg.solve(A,B));
        p1 = Polynomial(c)

        # plotting the input points
        # x = list(np.linspace(a,b,100))
        # y = list(map(f,x))
        # plt.plot(x,y,'r',linestyle = 'dashed',dashes=[4,2],linewidth='3',label = "Input function" )

        # plotting the computed polynomial
        # plt.title("Best Fit Polynomial: \n{}".format(p1.getPolynomialStr()))
        # p1.getplot(a,b)
        # plt.grid()
        # plt.xlabel("x")
        # plt.ylabel("f(x)")
        # plt.legend()
        # plt.show()
        return p1
    
    """
        function to compute the roots of the polynomial within a certain error using Aberth method
    """
    @handleException
    def printRoots(self,epsilon,selectReal):
        #calculating degree of the polynomial
        degree  = len(self.coeff)-1

        # computing the upper and lower bounds on the roots
        upperBound = 1+ 1/abs(self.coeff[-1]) * (max(abs(self.coeff[x]) for x in range(degree)))
        lowerBound = abs(self.coeff[0])/(abs(self.coeff[0]) + max(abs(self.coeff[x]) for x in range(1, degree + 1)))

        # stores the roots of the polynomial given
        rootsLs = []

        # we pick n (=degree) distinct numbers in complex plane within the lower and upper bounds 
        for _ in range(degree):
            r = random.uniform(lowerBound,upperBound)
            if(selectReal== 0):
                theta = random.uniform(0,2*np.pi)
                root = complex(r*np.cos(theta),r*np.sin(theta))
            
            # case when we only have to select real roots 
            else:
                root = r
            
            # append the number in rootsLs
            rootsLs.append(root)
        
        # computing the polynomial for the derivative of given polynomial
        d = self.derivative()

        # According to Newton Raphson self[root]/d[root] gives the error in the estimate of the root
        
        while(1):
            converged = 1
            for root in rootsLs:
                # checking if errors for all root values are within the given error limit
                # case when complex numbers are allowed as a root
                if( not selectReal and (self[root]/d[root]).real > epsilon):
                    converged = 0
                    break
                # case whe we only consider real number as the root
                elif(selectReal and (self[root]/d[root])>epsilon):
                    converged = 0
                    break

            # if all roots are within the max error allowed, we stop the iteration, else we continue   
            if(converged == 1):
                return rootsLs
            
            # Using Aberth Method to find the next approximation of the roots
            w = []
            for k in range(len(rootsLs)):
                #computing the offset numbers wₖ 
                numerator = self[rootsLs[k]]/ d[rootsLs[k]]
                sumRoots = 0
                for j in range(len(rootsLs)):
                    if(j!=k):
                        sumRoots += 1/(rootsLs[k]-rootsLs[j])
                denominator = 1-(numerator*sumRoots)
                wk = numerator/denominator
                w.append(wk)

            # Next set of approximations of roots of p(x) is given by:
            for i in range(degree):
                rootsLs[i] -= w[i]

"""
    Function takes an array of real number as argument and comutes a polynomial
    and outputs its roots within an error of 10⁻³
"""
def AlberthMethod(nums):
    # computing the polynomial g(x) = (x-a1)(x-a2)...(x-an)
    p = Polynomial([1])
    for ai in nums:
        p = p*Polynomial([-ai,1])
    
    # max error = epsilon
    epsilon= 0.001

    # Finding the equation of above polynomial
    print("Roots of the Polynomial:")
    print(p.getPolynomialStr())

    #computing the roots of the above polynomial and printing it
    roots= p.printRoots(epsilon,selectReal=0)
    print("are:")
    
    for _ in roots:
        print(_)


if __name__ == "__main__":
    #test case
    AlberthMethod([1,3,5,7,9])