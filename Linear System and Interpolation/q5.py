import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.interpolate  as interP

# evaluates the given function for x points
def getValue(x):
    y = (np.tan(x) * np.sin(30 * x)) * np.exp(x)
    return y

fig,axes = plt.subplots()

#plotting value of true function for 1000 points
x = np.linspace(0,1,1000)
y = [getValue(xi) for xi in x]
plt.plot(x,y,'b',label = "True")

#setting the graph axes
axes.set_xlabel("x")
axes.set_ylabel("f(x)")
axes.set_ylim(-4,4)
axes.set_xlim(0,1)


# inializing the curves for CubicSpline, Akima, Barycentric interpolating functions
cubicSpline, = plt.plot([],[],c='r',label="Cubic Spline")
akima,  = plt.plot([],[],'g',label = 'Akima')
barycentric,  = plt.plot([],[],'purple',label = "Barycentric")


# this function will be called each time, next value in frame is given through first argument of the function
def animate(frame):
    
    axes.set_title("Different interpolations of function for sample points between {}".format(frame/5))

    xpt = np.linspace(-3,3,frame)
    ypt = [getValue(xi) for xi in xpt]
    
    #update the curves with the corresponding interpolated values for cur frame
    cs = interP.CubicSpline(xpt,ypt)
    a = interP.Akima1DInterpolator(xpt,ypt)
    b = interP.BarycentricInterpolator(xpt,ypt)
    cubicSpline.set_data(x,cs(x))
    akima.set_data(x,a(x))
    barycentric.set_data(x,b(x))


    return cubicSpline, akima, barycentric

# points to be sampled is passed to function and every frame of the animation
frames = np.arange(5,170,5)
animation = animation.FuncAnimation (fig, animate, frames = frames, blit = True)
animation.save('q5.gif', fps = 5);

plt.legend(loc="upper left")
plt.grid()
plt.show()