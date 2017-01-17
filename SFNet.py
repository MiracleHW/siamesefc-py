#!/usr/bin/python2
import tensorflow as tf
import scipy.io as sio
import re
import numpy as np
import cv2
import os
from PIL import Image

class SFNet:
    params = []

    def __init__(self):
        self.params=self.GetParamFrMat()

    def conv2d(self,x, w, b, strides=1):
        if b==0:
            return tf.nn.conv2d(x,w,strides=[1, strides, strides, 1],padding='VALID')
        else:
            # Conv2D wrapper, with bias and relu activation
            return tf.nn.bias_add(tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='VALID'), b)


    def maxpool(self,x, size, strides):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, size, size, 1], strides=[1, strides, strides, 1], padding='VALID')

    # BatchNorm layer function
    def batchnorm(self,x, mean, variance, scale, offset):
        return tf.nn.batch_normalization(x, mean=mean, variance=variance, offset=offset, scale=scale,
                                         variance_epsilon=0.000001)

    # Relu layer function
    def relu(self,x):
        return tf.nn.relu(x)

    # get parameter from .mat file into parms(dict)
    def GetParamFrMat(self):
        params = {}  # define the block param dict to save params
        matpath = "./2016-08-17.net.mat"  # path to load .mat file

        netparams = sio.loadmat(matpath)["net"]["params"][0][0]
        i = 0
        length = netparams.size
        while (i < length):
            data = netparams[0][i]
            name = data["name"][0]
            x = data["value"].shape[0]
            if re.match("bn.x", name):
                value = tf.convert_to_tensor(data["value"])
                m, v = tf.split(1, 2, value)
                params[name + "m"] = tf.reshape(m, [x])
                params[name + "v"] = tf.reshape(v, [x])
            else:
                value = tf.convert_to_tensor(data["value"])
                if re.match("conv.f", name):
                    params[name] = value
                else:
                    value = tf.reshape(value, [x])
                    params[name] = value
            i += 1
        return params

    # simase-fc network
    def FCnet(self,input,size):
        input = tf.convert_to_tensor(input)
        input = tf.to_float(tf.reshape(input, [-1, size, size, 3]))

        # conv layer1
        conv1 = self.conv2d(input, w=self.params["conv1f"], b=self.params["conv1b"], strides=2)
        bn1 = self.batchnorm(conv1, mean=self.params["bn1xm"], variance=tf.square(self.params["bn1xv"]), scale=self.params["bn1m"],
                        offset=self.params["bn1b"])
        relu1 = self.relu(bn1)

        # max pooling layer1
        pool1x = self.maxpool(relu1, size=3, strides=2)
        pool1, x = tf.split(3, 2, pool1x)

        # conv layer2
        conv2f1, conv2f2 = tf.split(3, 2, self.params["conv2f"])
        conv2b1, conv2b2 = tf.split(0, 2, self.params["conv2b"])
        conv21 = self.conv2d(pool1, w=conv2f1, b=conv2b1)
        conv22 = self.conv2d(x, w=conv2f2, b=conv2b2)
        conv2 = tf.concat(3, [conv21, conv22])

        bn2 = self.batchnorm(conv2, mean=self.params["bn2xm"], variance=tf.square(self.params["bn2xv"]), scale=self.params["bn2m"],
                        offset=self.params["bn2b"])
        relu2 = self.relu(bn2)

        # max pooling layer2
        pool2 = self.maxpool(relu2, size=3, strides=2)

        # conv layer3
        conv3 = self.conv2d(pool2, w=self.params["conv3f"], b=self.params["conv3b"])
        bn3 = self.batchnorm(conv3, mean=self.params["bn3xm"], variance=tf.square(self.params["bn3xv"]), scale=self.params["bn3m"],
                        offset=self.params["bn3b"])
        relu3 = self.relu(bn3)
        relu3, x = tf.split(3, 2, relu3)

        # conv layer4
        conv4f1, conv4f2 = tf.split(3, 2, self.params["conv4f"])
        conv4b1, conv4b2 = tf.split(0, 2, self.params["conv4b"])
        conv41 = self.conv2d(relu3, w=conv4f1, b=conv4b1)
        conv42 = self.conv2d(x, w=conv4f2, b=conv4b2)
        conv4 = tf.concat(3, [conv41, conv42])

        bn4 = self.batchnorm(conv4, mean=self.params["bn4xm"], variance=tf.square(self.params["bn4xv"]), scale=self.params["bn4m"],
                        offset=self.params["bn4b"])
        relu4 = self.relu(bn4)
        relu4, x = tf.split(3, 2, relu4)

        # conv layer5
        conv5f1, conv5f2 = tf.split(3, 2, self.params["conv5f"])
        conv5b1, conv5b2 = tf.split(0, 2, self.params["conv5b"])
        conv51 = self.conv2d(relu4, w=conv5f1, b=conv5b1)
        conv52 = self.conv2d(x, w=conv5f2, b=conv5b2)
        conv5 = tf.concat(3, [conv51, conv52])

        #dict = {'conv1': conv1, 'bn1': bn1, 'relu1': relu1, 'pool1': pool1x, 'conv2': conv2, 'bn2': bn2, 'relu2': relu2,'conv3': conv3, 'z_features': conv5}
        return conv5

    #define the default congfig of GPU choice and memery usage
    def sessRun(self,input,feed_dict):
        #choose the gpu and usage of gpu memory
        os.environ['CUDA_VISIBLE_DEVICES'] = '2'
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

        result=sess.run(input,feed_dict=feed_dict)
        return result

    #cross-correlation
    def xcorr(self,input1,input2):
        map=self.conv2d(input1,input2,0)
        map=tf.mul(map,0.001)
        map=tf.add(map,-2.1483638)
        #map=tf.sigmoid(map)
        return map

    def eval_scoreMap(self,examplar,instance,e_size,i_size):
        e_input=tf.placeholder("float",shape=[e_size,e_size,3])
        i_input=tf.placeholder("float",shape=[None,i_size,i_size,3])

        e_fmap=self.FCnet(e_input,e_size)
        i_fmap=self.FCnet(i_input,i_size)

        shape=tf.shape(e_fmap)
        e_fmap=tf.reshape(e_fmap,[shape[1],shape[2],shape[3],-1])

        scoreMap=self.sessRun(self.xcorr(i_fmap,e_fmap),feed_dict={e_input:examplar,i_input:instance})
        return scoreMap

'''
img1=Image.open('/workspace/hw/WorkSpace/siamese-fc-py/data/BlurBody/img/0001.jpg')
img1=np.array(img1)

img2=cv2.imread( '/workspace/hw/WorkSpace/siamese-fc-py/data/BlurBody/img/0001.jpg')
img2=cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

print np.sqrt(np.sum(np.square(img1-img2)))
'''
'''
output=FCnet(img)

paths=['conv1','bn1','relu1','pool1','conv2','bn2','relu2','conv3','result']
dict={}
for path in paths:
    p='/disk2/huwei/SourceCode/siamese-fc/tracking/'+path+'.mat'
    if path=='result':
        path='z_features'
    mconv1 = tf.convert_to_tensor(sio.loadmat(p)["r"][path][0][0])
    x,y,z=sio.loadmat(p)["r"][path][0][0].shape
    mconv1 = tf.reshape(mconv1, [1, x, y, z])
    sub = tf.sub(output[path], mconv1)
    variance = tf.sqrt(tf.reduce_mean(tf.square(sub)))
    dict[path]=variance
sess=tf.Session()
result=sess.run(dict)
for key in dict:
    print key,":",result[key]
'''
'''
mconv1=tf.convert_to_tensor(sio.loadmat("/disk2/huwei/SourceCode/siamese-fc/tracking/result.mat")["r"]["z_features"][0][0])
mconv1=tf.reshape(mconv1,[1,35,50,256])

sub=tf.sub(output,mconv1)
variance=tf.sqrt(tf.reduce_mean(tf.square(sub)))

sess=tf.Session()
result=sess.run(variance)
print result
'''