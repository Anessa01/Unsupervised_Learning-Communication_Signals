
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

PI=3.1415926
N=128.
print(np.cos(PI))
def savepic(x,y1,y2):
    fig = plt.figure(figsize=(8,2.5))
    left, bottom, width, height = 0.1,0.1,0.8,0.8
    ax = fig.add_axes((left, bottom, width, height))
    ax.plot(x,y1)
    ax.plot(x,y2)
    ax.set_xlabel('t')
    ax.set_ylabel('y')
    fig.savefig('data/saved.png')
def DFT(y1,y2):
    Y1_R=np.zeros(128)
    Y1_I=np.zeros(128)
    Y2_R=np.zeros(128)
    Y2_I=np.zeros(128)
    for k in range(128):
        for n in range(128):
            Y1_R[k]+=y1[n]*np.cos(2.*PI*k*n/N)
            Y1_I[k]+=-y1[n]*np.sin(2.*PI*k*n/N)
            Y2_R[k]+=y2[n]*np.sin(2.*PI*k*n/N)
            Y2_I[k]+=y2[n]*np.cos(2.*PI*k*n/N)
    return Y1_R+Y2_R,Y1_I+Y2_I


f=open('2016.04C.multisnr.pkl','rb')
f.seek(0)
data=pickle.load(f,encoding='latin1')
print(data.keys())
data_QPSK=data[('QPSK',16)]
print(data_QPSK.shape)
print(data_QPSK[0,0])
i=50
y1=data_QPSK[i,0]
y2=data_QPSK[i,1]
x=range(128)

plt.title('QPSK'+str(i))
#y1,y2=DFT(y1,y2)
plt.stem(x, y1,use_line_collection=True)
plt.plot(x,y2)
plt.xlabel('t')
plt.ylabel('y')


#savepic(x,y1,y2)

plt.show()