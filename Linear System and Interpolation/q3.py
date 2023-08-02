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

class RowVectorFloat():
    """
        method to check the type of elements passed in the rowVector
    """
    @handleException
    def checkType(self,ls):
        for i in ls:
            if(not ((isinstance(i,int)) or isinstance(i,float))):
                return False;
        
        return True

    @handleException
    def __init__(self,ls):
        if(isinstance(ls,list)):
            if(self.checkType(ls)):
                # if input is valid create a list of the vector passed
                self.r = list(ls);
            else:
                raise Exception("Invalid Type- needs to be a row of type float or integer")
        else:
            raise Exception("Invalid Input");
    

    @handleException
    def __str__(self):
        # returns the list in the string form
        s = "";
        for i in self.r:
            s+=str(i)+"  "
        return s;

    @handleException
    def __len__(self):
        #returns len of the rowVectorFloat
        return len(self.r)

    @handleException
    def __getitem__(self,i):
        # returns the ith element of the list
        if(i>=len(self.r)) or (i<0):
            raise Exception("Index out of range")
        if not isinstance(i,int):
            raise Exception("Invalid Index- index should be of type int")

        return self.r[i];

    @handleException
    def __setitem__(self,index,value):
        # sets the (index)th value of list = value passed

        if(index>=len(self.r)) or (index<0):
            raise Exception("Index out of range")
        if not isinstance(index,int):
            raise Exception("Invalid Index- index should be of type int")

        self.r[index] = value;
    
    @handleException
    def __rmul__(self,scalar):
        # function to support right side multiplication of rowVector by a scalar
        return self.__mul__(scalar);

    @handleException  
    def __mul__(self,scalar):
        if not isinstance(scalar,(float,int)):
            raise Exception("Invalid scalar type")
        
        # multiples each element of rowVector by the scalar and returns an object of class rowVectorFloat
        r2 = [round((i*scalar),2) for i in self.r]
        return RowVectorFloat(r2);

    def __sub__(self,r2):
        #function to support subtraction between 2 row vectors
        if(len(self.r) != len(r2)):
            raise Exception("Lengths of both the objects of RowVectorFloat do not match")
        if(not isinstance(r2,RowVectorFloat)):
            raise Exception("r2 is not of type RowVectorFloat")
        else:
            r3=[round((self.r[i]-r2[i]),2)for i in range(len(self.r))]
        return RowVectorFloat(r3);

    
    @handleException
    def __add__(self,r2):
        # function to add 2 row vectors
        if(len(self.r) != len(r2)):
            raise Exception("Length of both the objects of RowVectorFloat do not match")
        if(not isinstance(r2,RowVectorFloat)):
            raise Exception("r2 is not of type RowVectorFloat")
        else:
            r3=[self.r[i]+r2[i] for i in range(len(self.r))]
        return RowVectorFloat(r3);

"""
    SquareMatrixFloat is a list of RowVectorFloat objects
"""
class SquareMatrixFloat(RowVectorFloat):
    @handleException
    def checkType(self,n):
        # check type of each element in the matrix 
        if(not (isinstance(n,int) or isinstance(n,float))):
            return False;
        return True

    @handleException
    def __init__(self,n):
        if(not self.checkType(n)):
            raise Exception("Invalid type of n given");
        else:
            # initialize the square matrix as a zero matrix of dim n X n
            self.n = n;
            self.matrix = [RowVectorFloat(list([0]*n)) for i in range(n)];
    

    @handleException
    def __str__(self):
        # return matrix elements in string form
        s = "The matrix is: \n"
        for i in range(self.n):
            s += str(self.matrix[i]) + "\n"
        return s

    @handleException     
    def sampleSymmetric(self):
        # method to randomly sample a symmetric matrix
        for i in range(self.n):
            for j in range(i,self.n):
                # randomly choses a value uniformly between 0 and 1 for non-diagonal elements
                # rounded to 2, to avoid inaccuracy caused due to floating point precision in python
                v = round(random.uniform(0,1),2);
                if(i == j):
                    # if it is a diagonal element value is randomly chosen between (0, n)
                    v = round(random.uniform(0,self.n),2)
                    self.matrix[i][j]=v;
                else:
                    self.matrix[i][j]=v;
                    self.matrix[j][i]=v;
                    
    @handleException
    def toRowEchelonForm(self):
        # method to convert the matrix into its row-echelon form

        #initializing pivot row and pivot column index
        # pivot = self.matrix[pivotRow][pivotColumn]
        pivotRow = 0;
        pivotColumn=0;
        
        # We iterate over all rows and columns while converting matrix to row-echelon form, and ensure following conditions are met:
        # 1. For each row that does not contain entirely zeros, the first non-zero entry is 1
        # 2. For two successive (non-zero) rows, the leading 1 in the higher row is further left than the leading one in the lower row.

        while (pivotRow<self.n) and (pivotColumn<self.n):
            # finding first non zero entry in the pivot column, starting from pivot row, as rows before it would already be set
            firstNonZero = pivotRow
            found = 0

            for r in range(firstNonZero,self.n):
                if(self.matrix[r][pivotColumn] != 0):
                    #if a nonZero element is found in the pivot column update the found flag
                    found = 1;
                    firstNonZero = r;
                    break;

            # if firstNonZero row is not same as pivotRow, then we need to swap those 2 rows
            if(found):
                if(firstNonZero != pivotRow):
                    temp = self.matrix[pivotRow];
                    self.matrix[pivotRow]=self.matrix[firstNonZero];
                    self.matrix[firstNonZero]=temp;
                
                #if it equal to pivot row then we just have to modify its elements as per row echelon form
            
            # if no nonZeroRow element exist in that column we move to next column keeping the pivotrow same;
            else:
                pivotColumn +=1
                continue;            

            # Now our pivot would be set at correct row and column.
            # so now we just have to make this pivot element self.matrix[pivotRow][pivotColumn] as 1 and all other elements below it in that column as 0;

            self.matrix[pivotRow] = self.matrix[pivotRow]*(float(1/self.matrix[pivotRow][pivotColumn]));
            for i in range(pivotRow+1,self.n):
                if(self.matrix[i][pivotColumn] != 0):
                    self.matrix[i] = self.matrix[i]*(float(1/self.matrix[i][pivotColumn]))
                    self.matrix[i] = self.matrix[i] - self.matrix[pivotRow]
                    #assuming -0.0 == 0.0
                    # self.matrix[i][pivotColumn] = 0
           
           # moving to next pivot element
            pivotRow +=1
            pivotColumn +=1

    @handleException
    def isDRDominant(self):
        # method to check if the matrix is diagonally row dominant
        # means sum of all elements in a row is strictly lesser than the diagonal element of that row
        for i in range(self.n):
            sum = 0;
            for j in range(self.n):
                if(i == j):
                    continue;
                sum += self.matrix[i][j];
            if(self.matrix[i][i]<sum):
                return False;
        return True;
    
    @handleException
    def jSolve(self,b,m):
        # handling cases when b, m are not of desired types
        if not isinstance(b,list):
            raise Exception("Invalid input for b- Expected a list")
        for i in b:
            if not self.checkType(i):
                raise Exception("b should be a list of integers consisting of integers or floats ")
        if not isinstance(m,int) or m<=0:
            raise Exception("m should be a positive integer")
        
        # exception is raised when the matrix we consider is not diagonally row dominant
        try:
            if not self.isDRDominant():
                raise Exception
        except:
            print (Exception)
            print("Not solving because convergence is not guaranteed")
            return ("","")
        
        # stores values of x at the k-1 iteration
        xPrev = [0]*self.n;

        # stores values of x in the kth iteration
        x = [0]*self.n;
        # list to store the error at each iteration => ||Ax-b|| (norm 2), where x values are as per the kth iteration resp.
        error = []

        # converting the matrix to numpy array 
        A = []
        for i in range(self.n):
            r = []
            for j in range(self.n):
                r.append(self.matrix[i][j])
            A.append(r)
        
        A = np.array(A)
        
        # calculating values of x and then error at each iteration as per jacobis method
        for k in range(1,m+1):
            #for each iteration k
            for i in range(self.n):
                #for each xi of current iteration
                sum = 0
                # sum(aij * xj) where i != j is found 
                for j in range(self.n):
                    if (i==j):
                        continue;
                    sum += self.matrix[i][j]*xPrev[j]
                # updating the value of xi in the curr iteration 
                x[i] = (b[i]-sum) / self.matrix[i][i];

            #after each iteration, calculating norm of error : ||Ax^(k)=b|| norm 2:
            ev = (np.matmul(A,np.array(x)) - np.array(b))
            e = np.linalg.norm(ev);
            error.append(e)

            #updating the xprev after each iteration
            xPrev = x
        
        #final x value: x
        return (error,x)



    @handleException
    def gsSolve(self,b,m):
        # handling cases when b, m are not of desired types
        if not isinstance(b,list):
            raise Exception("Invalid input for b- Expected a list")
        for i in b:
            if not self.checkType(i):
                raise Exception("b should be a list of integers consisting of integers or floats ")
        if not isinstance(m,int) or m<=0:
            raise Exception("m should be a positive integer")
    
        
        # stores values of x at the k-1 iteration
        xPrev = [0]*self.n;

        # stores values of x in the kth iteration
        x = [0]*self.n;
        # list to store the error at each iteration => ||Ax-b|| (norm 2), where x values are as per the kth iteration resp.
        error = []

        # converting the matrix to numpy array 
        A = []
        for i in range(self.n):
            r = []
            for j in range(self.n):
                r.append(self.matrix[i][j])
            A.append(r)
        
        A = np.array(A)
        
        # calculating values of x and then error at each iteration as per Gauss-Siedel method
        for k in range(1,m+1):
            #for each iteration k
            for i in range(self.n):
                #for each xi of current iteration
                sum = 0
                # sum(aij * xj) where i != j is found 
                for j in range(self.n):
                    if (i==j):
                        continue;
                    # if value of xj is not found in current iteration then we use its value from previous iteration
                    elif(j>i):
                        sum += self.matrix[i][j]*xPrev[j]
                    # else xj value is taken from current iteration itself
                    else:
                        sum += self.matrix[i][j]*x[j]

                x[i] = (b[i]-sum) / self.matrix[i][i];

            #after 1 iteration, calculating norm of error : ||Ax^(k)-b|| norm 2:
            ev = (A @ np.array(x)) - np.array(b)
            e = np.linalg.norm(ev);
            error.append(e)

            #updating the xprev after each iteration
            xPrev = x
        
        #final x value: x
        return (error,x)


"""
    function to visualize rate of convergence of Jacobi and Gauss-Siedel methods of a
linear system with a diagonally dominant square symmetric matrix.
"""
@handleException
def visualizeRateOfConvergence(sqMat):
    # exception is raised if matrix is not of class SquareMatrixFloat
    if not isinstance(sqMat,SquareMatrixFloat):
        raise Exception("Invalid input: to visualize rate of convergence- coefficient matrix should be of type SquareMatrixFloat")
    
    # Symmetric matrix is sampled till we get a matrix which is diagonally row Dominant
    while(1):
        sqMat.sampleSymmetric();
        if(sqMat.isDRDominant()):
            break
    
    #k = number of iterations
    k = 50

    #choosing a random vector b = [1,2,.....n]
    b = list([i for i in range(1,sqMat.n+1)])

    #calculating value of x and error using both jacobi and gauss-siedal method
    (jacobiError,jacobiX) = sqMat.jSolve(b,k);
    
    (gsError,gsX) = sqMat.gsSolve(b,k)
    x = list([i for i in range(1,k+1)])

    # plotting the graph for error values as given by jacobi method and gauss-siedel method for k iterations
    plt.plot(x,jacobiError,'b',label="Jacobi Method")
    plt.plot(x,gsError,'r',label="Gauss-Siedel Method")
    plt.title("Rate of Convergence of Jacobi and Gauss-Siedel method over {} iterations".format(k))
    plt.xlabel("Number of Iterations")
    plt.ylabel("Norm of error")
    plt.grid()
    plt.legend()
    plt.show()




if __name__ == "__main__":
    s = SquareMatrixFloat(4)
    visualizeRateOfConvergence(s)