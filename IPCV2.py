import numpy as np
import cv2
import tensorflow as tf
import skimage.color as color
import skimage.io as io
import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import os
from skimage.metrics import structural_similarity as ssim
import h5py
'''
list_images=glob.glob('D:\DIV2K_train_LR_x8\sample/*.png')
N=len(list_images)
data = np.zeros([N, 256, 256, 3]) # N is number of images for training

for i in range(N):
      data[i]=cv2.resize(cv2.imread(list_images[i]),(256,256))
num_train = N
Xtrain = color.rgb2lab(data[:num_train]/255)
xt = Xtrain[:,:,:,0]
yt = Xtrain[:,:,:,1:]
xt = xt.reshape(num_train, 256, 256, 1)
yt = yt.reshape(num_train, 256, 256, 2)
'''
def next_batch(num,data,labels):
      idx=np.arange(0,len(data))
      np.random.shuffle(idx)
      idx=idx[:num]
      data_shuffle=[data[i] for i in idx]
      labels_shuffle=[labels[i] for i in idx]
      return np.asarray(data_shuffle), np.asarray(labels_shuffle)

session = tf.Session()


x = tf.placeholder(tf.float32, shape = [None, 256, 256, 1], name = 'x')
ytrue = tf.placeholder(tf.float32, shape = [None, 256, 256, 2], name = 'ytrue')



def create_weights(shape):
      return tf.Variable(tf.random_normal(shape,stddev=0.1))
def create_bias(size):
      return tf.Variable(tf.constant(0.1, shape = [size]))

def convolution(inputs, num_channels, filter_size, num_filters, layer_name):
      weights = create_weights(shape = [filter_size, filter_size, num_channels, num_filters])
      bias = create_bias(num_filters)
      ## convolutional layer
      layer = tf.nn.conv2d(input = inputs, filter = weights, strides= [1, 1, 1, 1], padding = 'SAME', name=layer_name) + bias
      layer = tf.nn.relu(layer)
      return layer

def deconvolution(inputs, num_channels, filter_size, num_filters, layer_name):
      weights = create_weights(shape = [filter_size, filter_size, num_filters, num_channels])
      bias = create_bias(num_filters)
      ## deconvolutional layer
      input_shape = tf.shape(inputs)
      output_shapes = [input_shape[0],input_shape[1]*2,input_shape[2]*2,num_filters]
      layer = tf.nn.conv2d_transpose(value = inputs,output_shape = output_shapes, filter = weights, strides= [1, 2, 2, 1], padding = 'SAME', name=layer_name) + bias
      layer = tf.nn.relu(layer)
      return layer

def maxpool(inputs, kernel, stride):
      layer = tf.nn.max_pool(value = inputs, ksize = [1, kernel, kernel, 1], strides = [1, stride, stride, 1], padding = "SAME")
      return layer
def upsampling(inputs):
      layer = tf.image.resize_nearest_neighbor(inputs, (2*inputs.get_shape().as_list()[1], 2*inputs.get_shape().as_list()[2]))
      return layer

norm1 = tf.keras.layers.BatchNormalization()(x)
conv1 = convolution(norm1, 1, 3, 3,'conv1')
conv2 = convolution(tf.concat([conv1,norm1],3), 4, 3, 4, 'conv2')
conv3 = convolution(tf.concat([conv2,conv1,norm1],3), 8, 3, 8, 'conv3')
max1 = maxpool(conv3, 2, 2)

norm2 = tf.keras.layers.BatchNormalization()(max1)
conv4 = convolution(norm2, 8, 3, 4, 'conv4')
conv5 = convolution(tf.concat([conv4,norm2],3), 12, 3, 4,'conv5')
conv6 = convolution(tf.concat([conv5,conv4,norm2],3), 16, 3, 16, 'conv6')
max2 = maxpool(conv6, 2, 2)

norm3 = tf.keras.layers.BatchNormalization()(max2)
conv7 = convolution(norm3, 16, 3, 8, 'conv7')
conv8 = convolution(tf.concat([conv7,norm3],3), 24, 3, 8, 'conv8')
conv9 = convolution(tf.concat([conv8,conv7,norm3],3), 32, 3, 32, 'conv9')
max3 = maxpool(conv9, 2, 2)

norm4 = tf.keras.layers.BatchNormalization()(max3)
conv10 = convolution(norm4, 32, 3, 16, 'conv10')
conv11 = convolution(tf.concat([conv10,norm4],3), 48, 3, 16, 'conv11')
conv12 = convolution(tf.concat([conv11,conv10,norm4],3), 64, 3, 64, 'conv12')
max4 = maxpool(conv12, 2, 2)

norm5 = tf.keras.layers.BatchNormalization()(max4)
conv13 = convolution(norm5, 64, 3, 32, 'conv13')
conv14 = convolution(tf.concat([conv13,norm5],3), 96, 3, 32, 'conv14')
conv15 = convolution(tf.concat([conv14,conv13,norm5],3), 128, 3, 64, 'conv15')

deconv1 = deconvolution(conv15, 64, 3, 64, 'deconv1')
conv16 = convolution(tf.concat([deconv1,conv12],3), 128, 3, 64, 'conv16')
conv17 = convolution(conv16, 64, 3, 32, 'conv17')

deconv2 = deconvolution(conv17, 32, 3, 32, 'deconv2')
conv18 = convolution(tf.concat([deconv2,conv9],3), 64, 3, 32, 'conv18')
conv19 = convolution(conv18, 32, 3, 32, 'conv19')

deconv3 = deconvolution(conv19, 32, 3, 16, 'deconv3')
conv20 = convolution(tf.concat([deconv3,conv6],3), 32, 3, 16, 'conv20')
conv21 = convolution(conv20, 16, 3, 16, 'conv21')

deconv4 = deconvolution(conv21, 16, 3, 8, 'deconv4')
conv22 = convolution(tf.concat([deconv4,conv3],3), 16, 3, 8, 'conv22')
W_23=tf.Variable(tf.truncated_normal(shape=[3,3,8,2],stddev=0.1))
b_23=tf.Variable(tf.constant(0.1,shape=[2]))
conv23 = tf.nn.conv2d(conv22, filter = W_23, strides= [1, 1, 1, 1], padding = 'SAME', name='conv23') + b_23




loss = tf.losses.mean_squared_error(labels = ytrue, predictions = conv23)
cost = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(cost)
session.run(tf.global_variables_initializer())

saver=tf.train.Saver()

saver.restore(session,r'C:\Users\COPY\Documents\ipcv6.ckpt')

with h5py.File(r'C:\col\sample\big_dataset.hdf5', 'r') as f:
      xt = f['l/l_data']
      yt = f['ab/ab_data']
      num_epochs = 100
      for l in range(num_epochs):
            xb,yb = next_batch(200,xt,yt)
            session.run(optimizer, feed_dict = {x: xb, ytrue:yb})
            lossvalue = session.run(cost, feed_dict = {x: xb, ytrue:yb})
            print("epoch: " + str(l) + " loss: " + str(lossvalue))
            if ((l+1) == 100):
                  saver.save(session,r'C:\Users\COPY\Documents\ipcv7.ckpt')
            if ((l+1) == 200):
                  saver.save(session,r'C:\Users\COPY\Documents\ipcv8.ckpt')
            if ((l+1) == 300):
                  saver.save(session,r'C:\Users\COPY\Documents\ipcv9.ckpt')
            if ((l+1) == 400):
                  saver.save(session,r'C:\Users\COPY\Documents\ipcv10.ckpt')
            if ((l+1) == 500):
                  saver.save(session,r'C:\Users\COPY\Documents\ipcv11.ckpt')



      saver.save(session,r'C:\Users\COPY\Documents\ipcv.ckpt')



      #calculating validation loss
      
      valid_images=glob.glob('C:\col\DIV2K_valid_LR_bicubic_X2/*.png')
      Nv=len(valid_images)
      data_v = np.zeros([Nv, 256, 256, 3]) 

      for k in range(Nv):
            data_v[k]=cv2.resize(cv2.imread(valid_images[k]),(256,256))


      X_valid = color.rgb2lab(data_v[:Nv]*1.0/255)
      xv = X_valid[:,:,:,0]
      xv = xv.reshape(Nv, 256, 256, 1)
      yv = X_valid[:,:,:,1:]
      yv = yv.reshape(Nv, 256, 256, 2)

      output = session.run(conv23, feed_dict = {x: xv})

      rgb_valid=np.zeros([Nv,256,256,3])
      lab_valid=np.zeros([Nv,256,256,3])
      lab_valid[:,:,:,0]=xv.reshape([Nv,256,256])
      lab_valid[:,:,:,1:]=output
      for i in range(Nv):
            rgb_valid[i]=color.lab2rgb(lab_valid[i])


      loss_ssim=ssim(data_v/255,rgb_valid,multichannel=True,data_range=1)


      print('validation loss : ',loss_ssim)


      #print validation output

      comparison=np.zeros([256,768,3])
      comparison[:,:256,0]=xv[13].reshape([256,256])
      comparison[:,:256]=color.lab2rgb(comparison[:,:256])
      comparison[:,256:512,:]=rgb_valid[13]
      comparison[:,512:,:]=data_v[13]/255
      cv2.imshow('image',comparison)
      cv2.waitKey(0)
      '''

      output = session.run(conv17, feed_dict = {x: xt[8].reshape([1,256,256,1])})

      rgb_valid=np.zeros([256,256,3])
      lab_valid=np.zeros([256,256,3])
      lab_valid[:,:,0]=xt[8].reshape([256,256])
      lab_valid[:,:,1:]=output.reshape([256,256,2])
      rgb_valid=color.lab2rgb(lab_valid)

      comparison=np.zeros([256,512,3])
      comparison[:,:256,0]=xt[8].reshape([256,256])
      comparison[:,:256]=color.lab2rgb(comparison[:,:256])
      comparison[:,256:,:]=rgb_valid
      cv2.imshow('image',comparison)
      cv2.waitKey(0)
      '''

      

session.close()