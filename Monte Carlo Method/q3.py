import matplotlib.pyplot as plt
import random
import numpy as np
import math

# considering a unit length square with a circle inscribed in it,
# centre of the circle is at origin.
# circle's eqn => x^2 + y^2 = (0.5)^2

# this function estimates value of pi
# total number of points for simulation is given as input
def estimatePi(num):
    PtInCircle=0;
    PtInSquare=0;

    # xLs is list of all points from 1 to num
    xLs= np.linspace(0,num,num+1);

    # yLs stores 4 times the ratio of (number of points those were inside circle)/(number of points inside square) when i points were randomly generated, (simulation)
    yLs= []
    for i in range(num+1):
        # coordinates are randomly chosen
        x=random.uniform(-0.5,0.5);
        y=random.uniform(-0.5,0.5);
        dist = x**2 + y**2

        # if the coordinates are inside circle we increment PtInCircle Count
        if(dist<(0.5**2)):
            PtInCircle = PtInCircle+1;

        # every point in x range -0.5 to 0.5 and y range -0.5 to 0.5 lies within square
        PtInSquare = PtInSquare+1;

        #corresonding ratio is appended.
        ratio = PtInCircle/PtInSquare;
        yLs.append(4*ratio);

    #yls2 stores th actual pi value which is independent of simulation, thus x axis values
    yls2=[math.pi]*(num+1);

    #Plotting the graph between estimated and actual pi value
    plt.ylim(3.10,3.20);
    plt.grid();
    plt.plot(xLs,yLs,'b', xLs,yls2,'r');
    plt.title("Estimating using Monte Carlo Method");
    plt.ylabel("4 x fraction of points within circle");
    plt.xlabel("No of points generated");
    plt.legend(['Monte Carlo method', 'Value of math.pi'], loc='lower right');
    plt.show();

if __name__ == "__main__":
    estimatePi(2000000);