import numpy as np
import matplotlib.pyplot as plt
import random 

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from torch.utils.tensorboard import SummaryWriter




def create_curve(freq, ampl, freq_noise, ampl_noise, window_size, name):
    freq = random.gauss(freq, freq_noise) #add noise to frequency
    ampl = random.gauss(ampl, ampl_noise) #add noise to amplitude
    time = np.linspace(0, 10*np.pi, window_size)
    x = ampl*np.cos(freq*time)
    noise = np.random.normal(random.uniform(-1,0), random.uniform(0,1), window_size)
    x+=noise
    x = np.expand_dims(x, axis = 0) #expand to get 2d array (features, window length)
    x = np.expand_dims(x, axis = 0) #expand to get 3d array to store 2d elements
    #print(f"freq: {freq}, ampl:{ampl}", noise)
    fig = plt.figure()
    plt.plot(time, x[0,0,:])
    plt.title(f"{name}", fontsize=18)
    plt.xlabel('Time $\longrightarrow$', fontsize=18)
    plt.ylabel('Signal $\longrightarrow$', fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    fig.savefig(f'{name}', format='pdf')
    
    return x

#TEST
frequencies = [1,4,1.9,3.1]#[1,4,1.6,3.4]
amplitudes = [6,2,5,4]
freq_noise = 0.5
ampl_noise = 2
window_size = 1000
create_curve(frequencies[0], amplitudes[0], freq_noise, ampl_noise, window_size, "Source Domain: Class 0")
create_curve(frequencies[1], amplitudes[1], freq_noise, ampl_noise, window_size, "Source Domain: Class 1")
create_curve(frequencies[2], amplitudes[2], freq_noise, ampl_noise, window_size, "Target Domain: Class 0")
create_curve(frequencies[3], amplitudes[3], freq_noise, ampl_noise, window_size, "Target Domain: Class 1")