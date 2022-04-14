import  tensorflow as tf
import  numpy as np
from    tensorflow import keras
from    tensorflow.keras import Sequential, layers
from    PIL import Image
import  matplotlib
matplotlib.use('Agg')
from    matplotlib import pyplot as plt
import  pickle

def pkl2numpy(filename):
    f = open(filename, 'rb')
    f.seek(0)
    data = pickle.load(f, encoding='latin1')
    return data
def savepic(y,name):
    fig = plt.figure(figsize=(8,2.5))
    left, bottom, width, height = 0.1,0.1,0.8,0.8
    ax = fig.add_axes((left, bottom, width, height))
    x=range(88)
    y1=tf.reshape(y[:,0,:,:],[88])
    y2=tf.reshape(y[:,1,:,:],[88])
    ax.plot(x,y1)
    ax.plot(x,y2)
    ax.set_xlabel('t')
    ax.set_ylabel('y')
    fig.savefig('cdata/'+name+'.png')
def saveweight(w,name):
    fig = plt.figure(figsize=(8, 2.5))
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax = fig.add_axes((left, bottom, width, height))
    x = range(w.shape[0])
    y = tf.reshape(w, [w.shape[0]])
    ax.plot(x, y)
    ax.set_xlabel('n')
    ax.set_ylabel('w')
    fig.savefig('weight/' + name + '.png')
def savestem(h,name):
    fig = plt.figure(figsize=(8, 2.5))
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax = fig.add_axes((left, bottom, width, height))
    x = range(44)
    h1 = tf.reshape(h, [44])
    ax.stem(x, h1)
    ax.set_xlabel('n')
    ax.set_ylabel('h')
    fig.savefig('h_stem/' + name + '.png')

data=pkl2numpy("2016.04C.multisnr.pkl")
y1_test=data[('GFSK',18)][199,:,12:100]
for j in range(88):
    y1_test[0,j]=(y1_test[0,j]+2.5)/5.
    y1_test[1,j]=(y1_test[1,j]+2.5)/5.
y1_test=tf.reshape(y1_test,[1,2,88,1])
savepic(y1_test,"000")
model=tf.saved_model.load("./saved/1")

logits,h=model.call(y1_test)
y_hat=logits
#print(h)
#savepic(y1_test,'epoch'+str(epoch))
savepic(y_hat,"test")
savestem(h,"test")