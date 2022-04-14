import  os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import  tensorflow as tf
import  numpy as np
from    tensorflow import keras
from    tensorflow.keras import Sequential, layers
from    PIL import Image
import  matplotlib
#后台绘制图片
matplotlib.use('Agg')
from    matplotlib import pyplot as plt

import  pickle


#获取总数据集
f=open('2016.04C.multisnr.pkl','rb')
f.seek(0)
data=pickle.load(f,encoding='latin1')

R=22
tf.random.set_seed(R)
np.random.seed(R)
assert tf.__version__.startswith('2.')


def savepic(y1,y2,name):
    fig = plt.figure(figsize=(8,2.5))
    left, bottom, width, height = 0.1,0.1,0.8,0.8
    ax = fig.add_axes((left, bottom, width, height))
    x=range(128)
    y1=tf.reshape(y1,[128])
    y2=tf.reshape(y2,[128])
    ax.plot(x,y1)
    ax.plot(x,y2)
    ax.set_xlabel('t')
    ax.set_ylabel('y')
    fig.savefig('datas/'+name+'.png')


h_dim = 16
batchsz = 128
lr = 1e-3

y1_train=data[('GFSK',2)][:,1]
noise=tf.random.normal(y1_train.shape)
y1_train_noised=y1_train+noise

train_db=tf.data.Dataset.from_tensor_slices(y1_train_noised)
train_db = train_db.shuffle(batchsz * 5).batch(batchsz)
print(y1_train.shape)


def origin_database():
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train, x_test = x_train.astype(np.float32) / 255., x_test.astype(np.float32) / 255.
    # we do not need label
    train_db = tf.data.Dataset.from_tensor_slices(x_train)
    train_db = train_db.shuffle(batchsz * 5).batch(batchsz)
    test_db = tf.data.Dataset.from_tensor_slices(x_test)
    test_db = test_db.batch(batchsz)

    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)



class AE(keras.Model):

    def __init__(self):
        super(AE, self).__init__()
        #self.set_inputs([None,128])

        # Encoders
        self.encoder = Sequential([
            layers.Dense(64, activation=tf.nn.tanh),
            #layers.Conv2D(filters=4,kernel_size=(1,2),activation="relu",input_shape=(1,128),padding="valid"),
            layers.Dense(32, activation=tf.nn.tanh),
            layers.Dense(h_dim, activation=tf.nn.sigmoid)
        ])

        # Decoders
        self.decoder = Sequential([
            layers.Dense(32, activation=tf.nn.tanh),
            layers.Dense(64, activation=tf.nn.tanh),
            layers.Dense(128)
        ])


    def call(self, inputs, training=None):
        # [b, 784] => [b, 10]
        h = self.encoder(inputs)
        # [b, 10] => [b, 784]
        x_hat = self.decoder(h)

        return x_hat,h



model = AE()
model.build(input_shape=(None,128))
model.summary()

optimizer = tf.optimizers.Adam(lr=lr)

for epoch in range(150):

    for step, x in enumerate(train_db):

        #[b, 128] => [b, 128]
        #x = tf.reshape(x, [-1, 128])
        with tf.GradientTape() as tape:
            x_rec_logits ,h_rec= model(x)

            rec_loss=tf.losses.mean_squared_error(x,x_rec_logits)
            #rec_loss = tf.losses.binary_crossentropy(x, x_rec_logits, from_logits=True)
            rec_loss = tf.reduce_mean(rec_loss)
            #rec_loss = abs(rec_loss)

            loss_regularization = []
            l1_regularization=0
            for p in model.encoder.trainable_variables:
                loss_regularization.append(tf.nn.l2_loss(p))
                l1_regularization+=tf.reduce_sum(abs(p))
            for p in model.decoder.trainable_variables:
                loss_regularization.append(tf.nn.l2_loss(p))
                l1_regularization+=tf.reduce_sum(abs(p))
            loss_regularization = tf.reduce_sum(tf.stack(loss_regularization))

            rec_loss+=0.0001*loss_regularization
            rec_loss+=0.0001*l1_regularization

        grads = tape.gradient(rec_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))



        if step % 100 ==0:
            print(epoch, step, float(rec_loss))

    y1_test=data[('GFSK',8)][1246,0]
    y1_test=tf.reshape(y1_test,[1,128])
    logits,h=model(y1_test)
    y_hat=logits
    print(h)
    savepic(y1_test,y_hat,'epoch'+str(epoch))

model.save_weights('ae_model')

        # evaluation
        #x = next(iter(test_db))
        #logits = model(tf.reshape(x, [-1, 784]))
        #x_hat = tf.sigmoid(logits)
        # [b, 784] => [b, 28, 28]
        #x_hat = tf.reshape(x_hat, [-1, 28, 28])

        # [b, 28, 28] => [2b, 28, 28]
        #x_concat = tf.concat([x, x_hat], axis=0)
        #x_concat = x_hat
        #x_concat = x_concat.numpy() * 255.
        #x_concat = x_concat.astype(np.uint8)
        #save_images(x_concat, 'ae_images/rec_epoch_%d.png'%epoch)
