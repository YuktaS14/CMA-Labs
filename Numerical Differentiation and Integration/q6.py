import numpy as np
import matplotlib.pyplot as plt
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

    """
        Returns the coefficients of derivative polynomial
    """
    @handleException
    def derivative(self):
        dv = []
        # dv will store the coefficients of the derivative polynomial
        # coefficient will change as : a.xⁿ -> a.n.x⁽ⁿ⁻¹⁾ 
        # 0 th degree coefficient of the polynomial is omitted in its derivative
        for i in range(1,len(self.coeff)):
            d = i*self.coeff[i]
            dv.append(d)
        return dv

    """
        Calculates the coefficients of the integral polynomial
    """
    @handleException
    def integral(self):
        # 0th degree coefficient of this integral polynomial would be 0
        # and rest will change according to integral rule
        self.integralPoly = [0]
        for i in range(len(self.coeff)):
            v = self.coeff[i]/(i+1)
            self.integralPoly.append(v)

    """
        Returns the integral value at the given x
    """
    @handleException
    def integralValue(self,x):
        ans = 0;
        # finds integration value by substituting given x in the integral polynomial
        for i in range(len(self.integralPoly)):
            ans += self.integralPoly[i]* (x**i)
        return ans
    
    """
        Returns area under the curve for in the given interval
    """
    @handleException
    def area(self, a, b):
        if not isinstance(a,(float,int)) or not isinstance(b,(float,int)):
            raise Exception("Invalid type for interval given")

        if a>b:
            raise Exception("Invalid interval input: a should be less than b")
        # creates integral polynomial
        self.integral()
        # then finds integration value and thus area
        s = "Area in the interval [{},{}] is: {}".format(a, b, self.integralValue(b) - self.integralValue(a))
        return s 

if __name__ == "__main__":

    # test case 1:
    print("Test Case 1: ")
    p = Polynomial([1, 2, 3])
    pd = p.derivative()
    print(pd)

    # test case 2:
    print("\nTest Case 1: ")
    p = Polynomial([1, 2, 3])
    print(p.area(1,2))
