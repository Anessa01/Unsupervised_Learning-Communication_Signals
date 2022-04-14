import numpy as np

def self_relation(x):
    l = x.size
    y = np.zeros(l)
    for i in range(l):
        for j in range(l):
            y[i] += x[j] * x[(j + i) % l]
    return y


def power_spectrum(x):
    y = self_relation(x)
    PS = np.abs(np.fft.fftshift(np.fft.fft(y)))
    return PS
