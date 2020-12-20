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

#set random seeds
R=22
tf.random.set_seed(R)
np.random.seed(R)
assert tf.__version__.startswith('2.')
h_dim = 44
batchsz = 128
lr = 1e-3

#prepare datasets
data=pkl2numpy("2016.04C.multisnr.pkl")
y_train=data["QPSK",2][:,:,12:100]
for i in range(y_train.shape[0]):
    for j in range(88):
        y_train[i,0,j]=(y_train[i,0,j]+2.)/4.
        y_train[i,1,j]=(y_train[i,1,j]+2.)/4.
y_train=tf.reshape(y_train,[y_train.shape[0],2,88,1])
savepic(tf.reshape(y_train[200,:,:,:],[1,2,88,1]),"000")
train_db=tf.data.Dataset.from_tensor_slices(y_train)
train_db = train_db.shuffle(batchsz * 5).batch(batchsz)

print(y_train.shape)

class AE(keras.Model):

    def __init__(self):
        super(AE, self).__init__()

        #2*88->conv2D(1,2,40)->Dense(44)
        self.encoder1 = layers.Conv2D(filters=40, kernel_size=(1, 2), input_shape=(None,2, 88, 1), padding="valid")
        self.encoder2 = layers.Dense(44, activation=tf.nn.sigmoid)


        #Dense(44)->Dense(2*88)->Conv2D(1,1,81)->2*88
        self.decoder1 = layers.Dense(176, activation=tf.nn.sigmoid)
        #self.decoder2 = layers.Conv2D(filters=81, kernel_size=(1, 1), input_shape=(176,1), padding="valid")
        #self.pool1 = layers.MaxPool2D(pool_size=[1, 1], strides=1, padding='valid')


    def call(self, inputs, training=None):
        # [b, 2, 88] => [b, 44]
        h1 = self.encoder1(inputs)
        h2 = tf.reshape(h1,[-1,2*87*40])
        h3 = self.encoder2(h2)
        # [b, 44] => [b, 2, 88]
        x_hat1 = self.decoder1(h3)
        #x_hat2 = self.decoder2(x_hat1)
        #x_hat = self.pool1(x_hat2)
        x_hat=tf.reshape(x_hat1,[-1,2,88,1])
        return x_hat,h3

model = AE()
model.build(input_shape=(None,2,88,1))
model.summary()


optimizer = tf.optimizers.Adam(lr=lr)

for epoch in range(2000):

    for step, x in enumerate(train_db):


        with tf.GradientTape() as tape:
            x_rec_logits ,h_rec= model(x)

            rec_loss=tf.losses.mean_squared_error(x,x_rec_logits)
            #rec_loss = tf.losses.binary_crossentropy(x, x_rec_logits, from_logits=True)
            rec_loss = tf.reduce_mean(rec_loss)
            #rec_loss = abs(rec_loss)

        grads = tape.gradient(rec_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 ==0:
            print(epoch, step, float(rec_loss))

    y1_test=data[('QPSK',2)][200,:,12:100]
    y1_test=tf.reshape(y1_test,[1,2,88,1])
    logits,h=model(y1_test)
    y_hat=logits
    #print(h)
    #savepic(y1_test,'epoch'+str(epoch))
    savepic(y_hat,"epoch"+str(epoch))
    savetog(y_hat,y1_test,"epoch"+str(epoch))

