import matplotlib

from matplotlib import pyplot as plt


def savepic(y, slice, length, name):
    fig = plt.figure(figsize=(8, 2.5))
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax = fig.add_axes((left, bottom, width, height))
    if slice == 1:
        x = range(length)
        ax.plot(x, y)
    else:
        for i in range(slice):
            x = range(length)
            line = y[i, :]
            ax.plot(x, line)
    ax.set_xlabel('t')
    ax.set_ylabel('y')
    fig.savefig('saved/'+name+'.png')