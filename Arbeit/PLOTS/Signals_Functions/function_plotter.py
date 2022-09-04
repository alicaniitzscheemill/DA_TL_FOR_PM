from turtle import title
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

points = 100
a1 = 8.0
a2 = 8.0
N = 100
x1 = np.arange(0,N)-0.5*(N-1)
x2 = np.arange(0,N)

vec1 = signal.ricker(points, a1)
vec2 = signal.ricker(points, a2)
name = "Ricker Wavelet"
fig = plt.figure()
plt.plot(x1, vec1, label="0")
plt.plot(x2, vec2, label="50")
plt.title(f"{name}", fontsize=18)
plt.xlabel('Input $\longrightarrow$', fontsize=18)
plt.ylabel('Output $\longrightarrow$', fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
legend = plt.legend(loc="upper right", title="Scaling" + "Factor:", fontsize=15)
legend.set_title(title="Shifting Factor t:", prop={'size':14})
plt.tight_layout()
fig.savefig(f'{name}', format='pdf')

