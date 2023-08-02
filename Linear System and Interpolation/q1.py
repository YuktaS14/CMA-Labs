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
        r2 = [i*scalar for i in self.r]
        return RowVectorFloat(r2);

    
    @handleException
    def __add__(self,r2):
        # function to add 2 row vectors
        if(len(self.r) != len(r2)):
            raise Exception("Length of both the objects of RowVectorFloat do not match")
        if(not isinstance(r2,RowVectorFloat)):
            raise Exception("r2 is not of type RowVectorFloat")
        else:
            r3=[(self.r[i]+r2[i]) for i in range(len(self.r))]
        return RowVectorFloat(r3);
        
        

if __name__ == "__main__":

    # test case 1:
    print("For test Case 1: ")
    r = RowVectorFloat([1, 2, 4])
    print(r)

    #test case 2:
    print("\nfor test case 2: ")
    r = RowVectorFloat([1, 2 , 4])
    print(r[1])

    #test case 3:
    print("\nfor test case 3: ")
    r = RowVectorFloat([1,2,4])
    print(len(r))
    print(r[1])
    r[2] = 5
    print(r)

    #test case 4:
    print("\nfor test case 4: ")
    r1 = RowVectorFloat([1, 2 , 4])
    r2 = RowVectorFloat([1, 1 , 1])
    r3 = 2*r1 + (-3)*r2
    print(r3)