# 预训练对卷积调制识别网络的增益

## 卷积调制识别网络

> 1602.04105

CNN:

| Layer   | Output          | Weight         | Params     | Regul           |
| ------- | --------------- | -------------- | ---------- | --------------- |
| Input   | (1, 1, 2, 128)  | -              | -          | -               |
| Conv1   | (1, 64, 2, 126) | (64, 1, 1, 3)  | 192+64     | $\|W\|_2$       |
| Conv2   | (1, 16, 1, 124) | (16, 64, 2, 3) | 6144+16    | $\|W\|_2$       |
| Flatten | (1, 1984)       | -              | -          | -               |
| Dense1  | (1, 128)        | (1984, 128)    | 253952+128 | $\|\bold h\|_1$ |
| Dense2  | (1, 11)         | (128, 11)      | 1408+11    |                 |

Dropout: 0.5

Activation: ReLU for all layers except Softmax.



CNN2:

| Layer   | Output           | Weight          | Params      | Regul |
| ------- | ---------------- | --------------- | ----------- | ----- |
| Input   | (1, 1, 2, 128)   | -               | -           | -     |
| Conv1   | (1, 256, 2, 126) | (256, 1, 1, 3)  | 768+256     | -     |
| Conv2   | (1, 80, 1, 124)  | (80, 256, 2, 3) | 122880+80   | -     |
| Flatten | (1, 9920)        | -               | -           | -     |
| Dense1  | (1, 256)         | (9920, 256)     | 2539520+256 | -     |
| Dense2  | (1, 11)          | (11, 256)       | 2816+11     | -     |

Dropout: 0.5



DNN:(没做)

| Layer   | Output         | Weight     | Params     | Regul |
| ------- | -------------- | ---------- | ---------- | ----- |
| Input   | (1, 1, 2, 128) | -          | -          | -     |
| Flatten | (1, 256)       | -          | -          | -     |
| Dense1  | (1, 512)       | (256, 512) | 131072+512 | -     |
| Dense2  | (1, 256)       | (512, 256) | 131072+256 | -     |
| Dense3  | (1, 128)       | (256, 128) | 32768+128  | -     |
| Dense4  | (1, 11)        | (128, 11)  | 1408+11    | -     |

Dropout: 0.5

复现效果

![image-20210720170718815](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20210720170718815.png)

## 预训练方法（栈式自编码器）

> Why does Unsupervised Pre-training Help Deep Learning 

文中证明了栈式自编码器预训练的作用：优化初始参数空间（比随机初始化更靠近参数空间中的最优点 1.收敛更快 2.可以达到更高的精度）一定程度上可以视为一种正则化，提高泛化能力。

CNN2-pretrain:(AE applied on trainable layers)

Step 1:

| Layer   | Output           | Weight         | Params       | Trainable |
| ------- | ---------------- | -------------- | ------------ | --------- |
| Input   | (1, 1, 2, 128)   | -              | -            | -         |
| Conv1   | (1, 256, 2, 126) | (256, 1, 1, 3) | 768+256      | trainable |
| Flatten | (1, 64512)       | -              | -            | -         |
| Dense1  | (1, 256)         | (64512, 256)   | 16515072+256 | trainable |
| View    | (1, 2, 128)      | -              | -            | -         |

Step 2:

| Layer   | Output           | Weight          | Params      | Trainable |
| ------- | ---------------- | --------------- | ----------- | --------- |
| Input   | (1, 1, 2, 128)   | -               | -           | -         |
| Conv1   | (1, 256, 2, 126) | (256, 1, 1, 3)  | 768+256     | Frozen    |
| Conv2   | (1, 80, 1, 124)  | (80, 256, 2, 3) | 122880+80   | trainable |
| Flatten | (1, 9920)        | -               | -           | -         |
| Dense1  | (1, 256)         | (9920, 256)     | 2539520+256 | trainable |
| View    | (1, 2, 128)      | -               | -           | -         |

Step 3:

| Layer   | Output           | Weight          | Params      | Trainable |
| ------- | ---------------- | --------------- | ----------- | --------- |
| Input   | (1, 1, 2, 128)   | -               | -           | -         |
| Conv1   | (1, 256, 2, 126) | (256, 1, 1, 3)  | 768+256     | Frozen    |
| Conv2   | (1, 80, 1, 124)  | (80, 256, 2, 3) | 122880+80   | Frozen    |
| Flatten | (1, 9920)        | -               | -           | -         |
| Dense1  | (1, 256)         | (9920, 256)     | 2539520+256 | trainable |
| Dense2  | (1, 11)_-out_    | (11, 256)       | 2816+11     | trainable |
| Dense3  | (1, 256)         | (256, 11)       | 2816+256    | trainable |
| View    | (1, 2, 128)      | -               | -           | -         |

Then we get the pretrained paramset.

## 优化初始参数空间

采用完整标签，在随机初始化和预训练初始化条件下分别进行调制识别训练，（训练截止条件：测试准确率达到85%）

![image-20210720172855621](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20210720172855621.png)

在预训练条件下，模型只用了一个epoch就可以达到70%的准确率；同样自70%准确率开始，预训练条件下达到90%准确率所需时间也较短。

## 减少标签使用

采用少量样本和标签（10%）进行训练。

![image-20210720173030914](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20210720173030914.png)

![image-20210720173722282](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20210720173722282.png)

预训练过的网络保持了一定的泛化能力，对高信噪比信号的识别精度仍然能保持在70%到80%，而未经过预训练的参数则出现了明显的过拟合现象。

