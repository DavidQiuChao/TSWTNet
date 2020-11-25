'''
 An unofficial implement the network of
 "Burst Denoising via Temporally Shifted
 Wavelet Transform",ECCV2020 by Tensorflow.
'''

import copy
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim


'''
 The batch norm layer is from
 https://github.com/ChenGuangbinTHU/Tensorflow-BatchNorm
'''
def batch_normalization_layer(inputs,out_size,isTrain=True):
    eps = 0.001
    decay = 0.999
    pop_mean = tf.Variable(tf.zeros([out_size]),\
            trainable=False)
    pop_var = tf.Variable(tf.ones([out_size]),\
            trainable=False)
    scale = tf.Variable(tf.ones([out_size]))
    shift = tf.Variable(tf.zeros([out_size]))
    if isTrain:
        batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2])
        train_mean = tf.assign(pop_mean,\
                pop_mean*decay+batch_mean*(1-decay))
        train_var = tf.assign(pop_var,\
                pop_var*decay+batch_var*(1-decay))
        with tf.control_dependencies([train_mean,train_var]):
            return tf.nn.batch_normalization(inputs,\
                    batch_mean,batch_var,shift,scale,eps)
    else:
        return tf.nn.batch_normalization(inputs,pop_mean,\
                pop_var,shift,scale,eps)


def convBatchRelu(x,nf,isTrain):
    shape = x.get_shape().as_list()
    x = slim.conv2d(x,nf,[3,3])
    x = batch_normalization_layer(x,\
            nf,isTrain)
    x = tf.nn.relu(x)
    return x


def resTSUnit(x,buf,nf,isTrain,scope):
    with tf.name_scope(scope):
        c = x.get_shape().as_list()[3]
        x1,x2 = x[...,:c//8],x[...,c//8:]
        cov = tf.concat([buf,x2],3)
        cov = convBatchRelu(cov,nf,isTrain)
        cov = convBatchRelu(cov,nf,isTrain)
    return x+cov,x1


def resUnit(x,nf,isTrain,scope):
    with tf.name_scope(scope):
        conv = convBatchRelu(x,nf,isTrain)
        conv = convBatchRelu(conv,nf,isTrain)
    return x+conv


def convUnit(x,nf,isTrain,scope):
    with tf.name_scope(scope):
        conv = convBatchRelu(x,nf,isTrain)
        #conv = convBatchRelu(conv,nf,isTrain)
    return conv


def makeKernel(x,nf):
    x = np.repeat(np.expand_dims(x,-1),nf,-1)
    x = np.repeat(np.expand_dims(x,-1),nf,-1)
    return x


def getWavelet(nf):
    """wavelet decomposition using conv2d"""
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]
    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H
    kLL = makeKernel(harr_wav_LL,nf)
    kLH = makeKernel(harr_wav_LH,nf)
    kHL = makeKernel(harr_wav_HL,nf)
    kHH = makeKernel(harr_wav_HH,nf)
    kLL = tf.convert_to_tensor(kLL,dtype=tf.float32)
    kLH = tf.convert_to_tensor(kLH,dtype=tf.float32)
    kHL = tf.convert_to_tensor(kHL,dtype=tf.float32)
    kHH = tf.convert_to_tensor(kHH,dtype=tf.float32)
    return kLL,kLH,kHL,kHH


def wavePool(x,kLL,kLH,kHL,kHH,scope):
    with tf.name_scope(scope):
        ftLL = tf.nn.conv2d(x,kLL,\
                padding='SAME',strides=[1,2,2,1])
        ftLH = tf.nn.conv2d(x,kLH,\
                padding='SAME',strides=[1,2,2,1])
        ftHL = tf.nn.conv2d(x,kHL,\
                padding='SAME',strides=[1,2,2,1])
        ftHH = tf.nn.conv2d(x,kHH,\
                padding='SAME',strides=[1,2,2,1])
    return ftLL,ftLH,ftHL,ftHH


def waveUnpool(kLL,kLH,kHL,kHH,\
        ftLL,ftLH,ftHL,ftHH,\
        shape,scope):
    with tf.name_scope(scope):
        ftLL = tf.nn.conv2d_transpose(ftLL,kLL,\
                shape,strides=[1,2,2,1])
        ftLH = tf.nn.conv2d_transpose(ftLH,kLH,\
                shape,strides=[1,2,2,1])
        ftHL = tf.nn.conv2d_transpose(ftHL,kHL,\
                shape,strides=[1,2,2,1])
        ftHH = tf.nn.conv2d_transpose(ftHH,kHH,\
                shape,strides=[1,2,2,1])
    return ftLL+ftLH+ftHL+ftHH


def catWaveFeat(ftLL,ftLH,ftHL,ftHH,\
        hftLL,hftLH,hftHL,hftHH,nf,scope):
    with tf.name_scope(scope):
        ftLL = slim.conv2d(tf.concat([ftLL,hftLL],3),\
                nf,[1,1],scope=scope+'/ftLL')
        ftLH = slim.conv2d(tf.concat([ftLH,hftLH],3),\
                nf,[1,1],scope=scope+'/ftLH')
        ftHL = slim.conv2d(tf.concat([ftHL,hftHL],3),\
                nf,[1,1],scope=scope+'/ftHL')
        ftHH = slim.conv2d(tf.concat([ftHH,hftHH],3),\
                nf,[1,1],scope=scope+'/ftHH')
        return ftLL,ftLH,ftHL,ftHH


def network3D(input,bufs,isTrain=True):
    nf = 16
    idx = 0
    shape = input.get_shape().as_list()
    shape[3] = 2*nf
    shape2 = copy.deepcopy(shape)
    shape2[1] = shape2[1]/2
    shape2[2] = shape2[2]/2

    kLL,kLH,kHL,kHH = getWavelet(2*nf)

    # level 0
    conv = convUnit(input,2*nf,isTrain,'resU')

    # level 1
    tsu,bufs[idx] = resTSUnit(conv,bufs[idx],2*nf,\
            isTrain,'resTSU1')
    ftLL,ftLH,ftHL,ftHH =\
            wavePool(conv,kLL,kLH,kHL,kHH,'Pool1')

    # level 2
    hftLL,hftLH,hftHL,hftHH =\
            wavePool(tsu,kLL,kLH,kHL,kHH,'Pool2')
    idx += 1
    tsu,bufs[idx] = resTSUnit(tsu,bufs[idx],2*nf,\
            isTrain,'resTSU2')
    up = waveUnpool(kLL,kLH,kHL,kHH,\
        ftLL,ftLH,ftHL,ftHH,\
        shape,'Unpool2')
    idx += 1
    ftLL,bufs[idx] = resTSUnit(ftLL,bufs[idx],2*nf,\
            isTrain,'LLConv2')
    # do concatenation and 1x1 convolution
    tsu = tf.concat([tsu,up],3)
    tsu = slim.conv2d(tsu,2*nf,[1,1],scope='hCat2')
    ftLL,ftLH,ftHL,ftHH = catWaveFeat(ftLL,ftLH,ftHL,ftHH,\
        hftLL,hftLH,hftLH,hftHH,2*nf,'mCat2')

    # level 3
    hftLL,hftLH,hftHL,hftHH =\
            wavePool(tsu,kLL,kLH,kHL,kHH,'Pool3')
    idx += 1
    tsu,bufs[idx] = resTSUnit(tsu,bufs[idx],2*nf,\
            isTrain,'resTSU3')
    up = waveUnpool(kLL,kLH,kHL,kHH,\
        ftLL,ftLH,ftHL,ftHH,\
        shape,'Unpool3')
    lftLL,lftLH,lftHL,lftHH =\
            wavePool(ftLL,kLL,kLH,kHL,kHH,'Pool3_1')
    idx += 1
    ftLL,bufs[idx] = resTSUnit(ftLL,bufs[idx],2*nf,\
            isTrain,'LLConv3')
    # do concatenation and 1x1 convolution
    tsu = tf.concat([tsu,up],3)
    tsu = slim.conv2d(tsu,2*nf,[1,1],scope='hCat3')
    ftLL,ftLH,ftHL,ftHH = catWaveFeat(ftLL,ftLH,ftHL,ftHH,\
        hftLL,hftLH,hftLH,hftHH,2*nf,'mCat3')

    # level 4
    hftLL,hftLH,hftHL,hftHH =\
            wavePool(tsu,kLL,kLH,kHL,kHH,'Pool4')
    idx += 1
    tsu,bufs[idx] = resTSUnit(tsu,bufs[idx],2*nf,\
            isTrain,'resTSU4')
    up = waveUnpool(kLL,kLH,kHL,kHH,\
        ftLL,ftLH,ftHL,ftHH,\
        shape,'Unpool4')
    mftLL,mftLH,mftHL,mftHH =\
            wavePool(ftLL,kLL,kLH,kHL,kHH,'Pool4_1')
    idx += 1
    ftLL,bufs[idx] = resTSUnit(ftLL,bufs[idx],2*nf,\
            isTrain,'LLConv4')
    up_1 = waveUnpool(kLL,kLH,kHL,kHH,\
        lftLL,lftLH,lftHL,lftHH,\
        shape2,'Unpool4_1')
    ftLL = tf.concat([ftLL,up_1],3)
    idx += 1
    lftLL,bufs[idx] = resTSUnit(lftLL,bufs[idx],2*nf,\
            isTrain,'LLConv4_1')
    # do concatenation and 1x1 convolution
    tsu = tf.concat([tsu,up],3)
    tsu = slim.conv2d(tsu,2*nf,[1,1],scope='hCat4')
    ftLL,ftLH,ftHL,ftHH = catWaveFeat(ftLL,ftLH,ftHL,ftHH,\
        hftLL,hftLH,hftLH,hftHH,2*nf,'mCat4')
    lftLL,lftLH,lftHL,lftHH = catWaveFeat(lftLL,lftLH,lftHL,lftHH,\
        mftLL,mftLH,mftLH,mftHH,2*nf,'mCat4_1')

    # level 5
    up_1 = waveUnpool(kLL,kLH,kHL,kHH,\
        lftLL,lftLH,lftHL,lftHH,\
        shape2,'Unpool5_1')
    ftLL = tf.concat([ftLL,up_1],3)
    ftLL = slim.conv2d(ftLL,2*nf,[1,1],scope='hCat5_1')
    up = waveUnpool(kLL,kLH,kHL,kHH,\
        ftLL,ftLH,ftHL,ftHH,\
        shape,'Unpool5')
    idx += 1
    tsu,bufs[idx] = resTSUnit(tsu,bufs[idx],2*nf,\
            isTrain,'resTSU5')
    tsu = tf.concat([tsu,up],3)
    out = slim.conv2d(tsu,2*nf,[1,1],scope='hCat5')
    out = tf.concat([conv,out],3)
    out = slim.conv2d(out,4,[1,1],scope='output')
    #out += input
    return out,bufs


def network2D(input,isTrain=True):
    nf = 16
    idx = 0
    shape = input.get_shape().as_list()
    shape[3] = 2*nf
    shape2 = copy.deepcopy(shape)
    shape2[1] = shape2[1]/2
    shape2[2] = shape2[2]/2

    kLL,kLH,kHL,kHH = getWavelet(2*nf)

    # level 0
    conv = convUnit(input,2*nf,isTrain,'resU')

    # level 1
    tsu = resUnit(conv,2*nf,\
            isTrain,'resTSU1')
    ftLL,ftLH,ftHL,ftHH =\
            wavePool(conv,kLL,kLH,kHL,kHH,'Pool1')

    # level 2
    hftLL,hftLH,hftHL,hftHH =\
            wavePool(tsu,kLL,kLH,kHL,kHH,'Pool2')
    idx += 1
    tsu = resUnit(tsu,2*nf,\
            isTrain,'resTSU2')
    up = waveUnpool(kLL,kLH,kHL,kHH,\
        ftLL,ftLH,ftHL,ftHH,\
        shape,'Unpool2')
    idx += 1
    ftLL = resUnit(ftLL,2*nf,\
            isTrain,'LLConv2')
    # do concatenation and 1x1 convolution
    tsu = tf.concat([tsu,up],3)
    tsu = slim.conv2d(tsu,2*nf,[1,1],scope='hCat2')
    ftLL,ftLH,ftHL,ftHH = catWaveFeat(ftLL,ftLH,ftHL,ftHH,\
        hftLL,hftLH,hftLH,hftHH,2*nf,'mCat2')

    # level 3
    hftLL,hftLH,hftHL,hftHH =\
            wavePool(tsu,kLL,kLH,kHL,kHH,'Pool3')
    idx += 1
    tsu = resUnit(tsu,2*nf,\
            isTrain,'resTSU3')
    up = waveUnpool(kLL,kLH,kHL,kHH,\
        ftLL,ftLH,ftHL,ftHH,\
        shape,'Unpool3')
    lftLL,lftLH,lftHL,lftHH =\
            wavePool(ftLL,kLL,kLH,kHL,kHH,'Pool3_1')
    idx += 1
    ftLL = resUnit(ftLL,2*nf,\
            isTrain,'LLConv3')
    # do concatenation and 1x1 convolution
    tsu = tf.concat([tsu,up],3)
    tsu = slim.conv2d(tsu,2*nf,[1,1],scope='hCat3')
    ftLL,ftLH,ftHL,ftHH = catWaveFeat(ftLL,ftLH,ftHL,ftHH,\
        hftLL,hftLH,hftLH,hftHH,2*nf,'mCat3')

    # level 4
    hftLL,hftLH,hftHL,hftHH =\
            wavePool(tsu,kLL,kLH,kHL,kHH,'Pool4')
    idx += 1
    tsu = resUnit(tsu,2*nf,\
            isTrain,'resTSU4')
    up = waveUnpool(kLL,kLH,kHL,kHH,\
        ftLL,ftLH,ftHL,ftHH,\
        shape,'Unpool4')
    mftLL,mftLH,mftHL,mftHH =\
            wavePool(ftLL,kLL,kLH,kHL,kHH,'Pool4_1')
    idx += 1
    ftLL = resUnit(ftLL,2*nf,\
            isTrain,'LLConv4')
    up_1 = waveUnpool(kLL,kLH,kHL,kHH,\
        lftLL,lftLH,lftHL,lftHH,\
        shape2,'Unpool4_1')
    ftLL = tf.concat([ftLL,up_1],3)
    idx += 1
    lftLL = resUnit(lftLL,2*nf,\
            isTrain,'LLConv4_1')
    # do concatenation and 1x1 convolution
    tsu = tf.concat([tsu,up],3)
    tsu = slim.conv2d(tsu,2*nf,[1,1],scope='hCat4')
    ftLL,ftLH,ftHL,ftHH = catWaveFeat(ftLL,ftLH,ftHL,ftHH,\
        hftLL,hftLH,hftLH,hftHH,2*nf,'mCat4')
    lftLL,lftLH,lftHL,lftHH = catWaveFeat(lftLL,lftLH,lftHL,lftHH,\
        mftLL,mftLH,mftLH,mftHH,2*nf,'mCat4_1')

    # level 5
    up_1 = waveUnpool(kLL,kLH,kHL,kHH,\
        lftLL,lftLH,lftHL,lftHH,\
        shape2,'Unpool5_1')
    ftLL = tf.concat([ftLL,up_1],3)
    ftLL = slim.conv2d(ftLL,2*nf,[1,1],scope='hCat5_1')
    up = waveUnpool(kLL,kLH,kHL,kHH,\
        ftLL,ftLH,ftHL,ftHH,\
        shape,'Unpool5')
    idx += 1
    tsu = resUnit(tsu,2*nf,\
            isTrain,'resTSU5')
    tsu = tf.concat([tsu,up],3)
    out = slim.conv2d(tsu,2*nf,[1,1],scope='hCat5')
    #out = tf.concat([conv,out],3)
    out = slim.conv2d(out,4,[1,1],scope='output')
    out += input
    return out
