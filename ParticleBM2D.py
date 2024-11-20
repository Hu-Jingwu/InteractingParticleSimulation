import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy import linalg as LA
import scipy.stats as stats
import random

ssize = 100
diffusivity = 0.5
timestep = 0.015
sdrate = 4
vfrate = 0
agg =0
c_birth = 4

Sample = np.random.rand(2,ssize)


def animate(i):
    global Sample, CSample, jumprate, diffusivity, vfrate, agg#, infected, removed, counter, T, SCounts, ICounts, RCounts
    speed0= diffusivity*timestep
    Sample += np.random.normal(0, 1, size=(2, ssize))*speed0
    
    Sample = Sample%1
    Spoints.set_data(Sample)

    return Spoints



Title = 'Simulation: 2D Dimensional Bronian Motion'

fig, ax= plt.subplots()
fig.suptitle(Title, fontsize=16)
ax.set_aspect('equal')
ax.set_xlim(0,1);
ax.set_ylim(0,1);

Spoints, = ax.plot([], [], 'bo', markersize=5, lw=2)

ani = animation.FuncAnimation(fig, animate, frames=400, interval=10)
plt.get_current_fig_manager().window.state('zoomed')

plt.show()