import matplotlib.pyplot as plt
import numpy as np
import random
import math
import sys




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
        if not isinstance(p, Polynomial):
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
                s+= str(self.coeff[i])
                continue;

            if(self.coeff[i]>0):
                s+="+"
            if(self.coeff[i]<0):
                s+="-"
            if(abs(self.coeff[i]) != 1):
                s+= str(round(abs(self.coeff[i]),2))
            
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

        #plotting for 50 points
        x = list(np.linspace(xmin,xmax,50))
        y = list([self[xi] for xi in x])
        plt.plot(x,y,'b')

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
        
if __name__ == "__main__":

    # test case 1:
    print("For test case 1:")
    p = Polynomial([1, 2, 3])
    print(p)

    # test case 2:
    print("\nFor test case 2:")
    p1 = Polynomial([1, 2, 3])
    p2 = Polynomial([3, 2])
    p3 = p1 + p2
    print(p3)

    # test case 3:
    print("\nFor test case 3:")
    p1 = Polynomial([1, 2, 3])
    p2 = Polynomial([3, 2, 4, 1])
    p3 = p1 - p2
    print(p3)

    # test case 4:
    print("\nFor test case 4:")
    p1 = Polynomial([1, 2, 3])
    p2 = (-0.5)*p1
    print(p2)

     # test case 5:
    print("\nFor test case 5:")
    p1 = Polynomial([-1, 1])
    p2 = Polynomial([1, 1, 1])
    p3 = p1 * p2
    print(p3)

     # test case 6:
    print("\nFor test case 6:")
    p = Polynomial([1, 2, 3])
    print(p[2])

     # test case 7:
    print("\nFor test case 7:")
    p = Polynomial([1, -1, 1, -1])
    p.show(-1, 2)

     # test case 8:
    print("\nFor test case 8:")
    p = Polynomial([])
    p.fitViaMatrixMethod([(1,4), (0,1), (-1, 0), (2, 15), (3,12)])

     # test case 9:
    print("\nFor test case 9:")
    p = Polynomial([])
    p.fitViaLagrangePoly([(1,-4), (0,1), (-1, 4), (2, 4), (3,1)])