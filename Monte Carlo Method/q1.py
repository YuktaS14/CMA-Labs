import matplotlib.pyplot as plt
import math
import numpy  as np

# ls ith entry stores value log(i!)
# precomputation of these values reduces many computation (overlapping subproblems) required for different values of i
ls = [0];
for i in range(2,10**6+1):
    ls.append(ls[i-2]+math.log(i));

# x is the number from 1 to 10^6;
# y1 is log(x!)
x = np.linspace(1,int(10**6),int((10**6))).astype(int);
y1 = ls;

# x2 are few points from x taken so as to observe clearly the plotting of both graphs (to avoid overlapping of 2 plots)
# y2 is log(strirlings approximation for x!) = [x*(log(x)-1) + 0.5 * (log(2*pi*x))] .....(log x has base e)
x2 = np.linspace(1,int(10**6),30);
y2 = x2*(np.log(x2)-1) + 0.5*np.log(2*np.pi*x2)

# calculates sterlings approximation for all points in x
y3 = x*(np.log(x)-1) + 0.5*np.log(2*np.pi*x)
"""
   to plot the difference betweeen 2 functions uncomment following code 
"""
# plt.plot(x,y1-y3,'b');
# plt.xlabel("x");
# plt.ylim(-0.1,0.2);
# plt.ylabel("log(x!)- log(Sterling's approximation for x!)");
# plt.show();

# plotting the functions
plt.plot(x,y1,'r',x2,y2,'bo');
plt.xlabel("x");
plt.ylabel("f(x)");
plt.legend(['log (n!)', 'log (Stirling\'s Formula)'], loc= "lower right");
plt.show();
