import  os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)



#tensorflow
import  tensorflow as tf
import  numpy as np
from    tensorflow import keras
from    tensorflow.keras import Sequential, layers
from    PIL import Image
import  matplotlib

#后台绘制图片
matplotlib.use('Agg')
from    matplotlib import pyplot as plt

#读入数据集
import  pickle

#transfer .pkl file to numpy datas
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

def savetog(ya,yb,name):
    fig = plt.figure(figsize=(8, 2.5))
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax = fig.add_axes((left, bottom, width, height))
    x = range(88)
    y1 = tf.reshape(ya[:, 0, :, :], [88])
    y2 = tf.reshape(ya[:, 1, :, :], [88])
    y3 = tf.reshape(yb[:, 0, :, :], [88])
    y4 = tf.reshape(yb[:, 1, :, :], [88])
    ax.plot(x, y1)
    ax.plot(x, y2)
    ax.plot(x, y3)
    ax.plot(x, y4)
    ax.set_xlabel('t')
    ax.set_ylabel('y')
    fig.savefig('tdata/' + name + '.png')

#assert dir
assets_dir = './cdata'
if not os.path.isdir(assets_dir):
    os.makedirs(assets_dir)
#set random seeds
R=22
tf.random.set_seed(R)
np.random.seed(R)
assert tf.__version__.startswith('2.')
h_dim = 44
batchsz = 128
lr = 0.0005
l1r=0.0001
l2r=0.001

DLabel = {'8PSK':0, 'AM-DSB':1, 'AM-SSB':2, 'BPSK':3, 'CPFSK':4, 'GFSK':5, 'PAM4':6, 'QAM16':7, 'QAM64':8, 'QPSK':9, 'WBFM':10}

#prepare datasets
data=pkl2numpy("RML2016.10a_dict.pkl")
trainsz = 1000

def alloc(data):
    x_train = np.zeros((trainsz * 220, 2, 128))
    x_label = np.zeros(trainsz * 220)
    i = -1
    for [label, snr] in data:
        for j in range(0,trainsz):
            i += 1
            x_train[i] = data[label, snr][j]
            x_label[i] = DLabel[label]
        print(label, snr, " loaded")
    return x_train, x_label

def dynamicalloc(data):
    x_train = np.empty((0,2,128))
    x_label = np.empty((0))

    for [label,snr] in data:
        if snr > 10:
            x_train = np.append(x_train,np.reshape(data[label,snr],(trainsz,2,128)),axis = 0)
            x_label = np.append(x_label,np.full((trainsz),DLabel[label]), axis = 0)
            print(label,snr," loaded")
    return x_train, x_label

x_train, x_label = dynamicalloc(data)


# x_train(256*220, 2, 128)
# x_label(256*220)
x_train = tf.reshape(x_train,[x_train.shape[0],256])
labels = tf.constant([0,1,2,3,4,5,6,7,8,9,10])


print(x_train.shape, x_label.shape)
print(x_label[40])

model = tf.keras.Sequential([
    layers.Dense(128, activation='relu', kernel_initializer='he_normal', input_shape=(256,)),
    layers.Dense(64, activation='relu', kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.l1(0.01)),
    layers.Dense(32, activation='relu', kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    layers.Dense(11, activation='softmax')
])


model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, x_label, batch_size=128, epochs=1000, validation_split=0.3, verbose=0)

print(history.history['val_accuracy'])


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'], loc='upper left')
#plt.show()
plt.savefig('cdata/epoch10.png')


