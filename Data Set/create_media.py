# thesis drawings
from matplotlib import pyplot as plt
import numpy as np


def plot_piecewise_linear():
    def f(x):
        return 2**((x-98)/6)

    hs = [80+6*i for i in range(11)]
    for i,j in enumerate(hs[:-1]): 
            plt.plot([hs[i],hs[i+1]],[f(hs[i]),f(hs[i+1])],'ro-', lw=4, markersize=1,alpha=0.5)
    x = np.linspace(76,140,1000)
    plt.scatter(x,f(x),s=0.5,zorder=100)
    plt.show()


#plot_piecewise_linear()

def plot_hourly_load():
    x = [i for i in range(24)]
    y = [1.05,0.83,0.75,0.64,0.7,0.64,0.7,0.83,1.4,1.4,1.36,1.28,1.39,1.39,1.25,1.28,1.29,1.58,2,2,2,1.8,1.67,1.42]
    plt.xticks(x)
    plt.xlabel('Hour of the day')
    plt.ylabel('Relative Electricity Usage')
    plt.scatter(x,y)
    plt.show()

plot_hourly_load()
