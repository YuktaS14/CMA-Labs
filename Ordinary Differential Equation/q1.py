import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.integrate as integrate
from matplotlib.animation import FuncAnimation



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
            if(round(abs(self.coeff[i]),2) == 0.00):
                continue;
            if self.coeff[i] == 0:
                continue;
                
            if i == 0:
                s+= str(round(self.coeff[i],2))
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
        # plt.plot(x,y,'ro')

        # sorting values of x to get interval in which we need to plot the graph -> (a,b)
        x.sort()

        return p1

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
                raise Exception(f"{type(i[0])}, {type(i[1])} (x,y) should be of type float or int");
        

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
        c = list(np.linalg.solve(A,b))
  
        p1 = Polynomial(c)
        return p1


"""
    Function that uses forward Euler method to solve the ODE x'(t) = -2*x(t)
"""

def forwardEuler(f,diffEq,discrSteps,x0,t0,T):
    xls = []
    tls = []
    polys =[]

    #solution for differential equation is found for each discretization step h
    for h in discrSteps:
        xPoints = [x0]
        # generating t points with current step size
        tPoints = list(np.arange(t0,T+h,h))
        
        #computing xpoints from previous value of t and x using forward euler method 
        pointsLs = []
        for i in range(len(tPoints)-1):
            xNext = xPoints[i]+ h* diffEq(tPoints[i],xPoints[i])
            xPoints.append(xNext)
            pointsLs.append((float(tPoints[i]), float(xPoints[i])))
        
        #after we get the points, we find the polynomial of degree (len(pointsLs)-1), which best fits through the points
        p1 = Polynomial([0])
        p1 = p1.bestFitPolynomial(pointsLs,len(pointsLs)-1)
        
        #storing the solution for each discretization step, and plotting it later
        xls.append(xPoints)
        tls.append(tPoints)
        polys.append(p1)

        #printing the polynomial which best fits through the points
        print(f"for h = {h}, Polynomial equation is:")
        print(p1.getPolynomialStr(),"\n")

    # title and axes labels for the plot
    plt.title("Forward Euler Method to solve x'(t) = -2x(t)")
    plt.xlabel("t")
    plt.ylabel("x(t)")
    c = ['b','y',"r","g","m"]
    
    #plotting the polynomial and the points obtained for each discretization step
    for i in range(len(polys)):
        x = list(np.linspace(tls[i][0],len(tls[i])-1,100))
        y = list([polys[i][xi] for xi in x])
        plt.plot(x,y,c[i],label = f"h: {discrSteps[i]}")
        # plt.plot(tls[i],xls[i],c[i]+'o',label = f"h: {discrSteps[i]}")

    # plotting the actual function by taking points in the range t= [t0,T] and calculating their function value 
    tpts = list(np.linspace(t0,T,100))
    xpts = list(map(f,tpts))
    plt.plot(tpts,xpts,'k',label="Actual Function")

    #setting x,y axes limits
    plt.ylim(-8,8)
    plt.xlim(-1,10)
    plt.legend()
    plt.show() 

# f: x(t) = 5*e⁽⁻²*ᵗ⁾ 
def f(t):
    return 5*(math.e)**(-2*t)

# f': x'(t) = -2*(x(t))
def diffEq(t,x):
    return -2*x

if __name__ == "__main__":
    # finding the solution for the ode using following discretization steps and initial value of x= 5, t=0, in the interval t = [0,10]
    discrSteps = [0.1,0.5,1,2,3]
    forwardEuler(f,diffEq,discrSteps,5,0,10)


