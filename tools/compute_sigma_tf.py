# -*- encoding:utf8 -*-
'''
author: Jay Smelly
date: 2017.03.12
Reference: paper Reference Based LSTM for Image Captioning ∗ weighted training part
compute sigma
'''

import numpy as np
import tensorflow as tf

def loadImgFeatureNumpy(path):
    img_fs = np.load(path)
    return img_fs

def imgDistanceTensor(img_fs):
    A = tf.pack(img_fs)
    vecProd = tf.matmul(A, A, transpose_b = True)
    # print 'vecProdi:',vecProd.eval()
    SqA = tf.square(A)
    sumSqA = tf.reduce_sum(SqA, 1, keep_dims=True)
    # print sumSqA.get_shape()
    sumSqAEx = tf.tile(
            sumSqA,
            tf.pack([1, vecProd.get_shape()[1]]))
    # print 'sumSqAEx:', sumSqAEx.eval()
    sumSqBEx = tf.tile(
            tf.transpose(sumSqA), 
            tf.pack([vecProd.get_shape()[0], 1]))
    SqED = sumSqBEx + sumSqAEx - 2*vecProd
    # print 'SqED:',SqED.eval()
    ED = tf.sqrt(SqED)
    # tf.add_check_numerics_ops()
    # print 'ED:', ED.eval()
    return ED
    
def imgDistanceMatrix(img_fs):
    A = np.matrix(img_fs)
    BT = np.matrix(img_fs).transpose()
    vecProd = A * BT
    SqA =  A.getA()**2
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))    
    SqB = A.getA()**2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))    
    SqED = sumSqBEx + sumSqAEx - 2*vecProd   
    ED = np.sqrt(SqED)
    return np.array(ED)

def sigmaTensor(imgDis):
    length = imgDis.get_shape().as_list()[0]
    denominator = tf.divide(tf.multiply(length, length-1), 2)
    nominator = tf.divide(tf.reduce_sum(imgDis), 2)
    # print imgDis.eval()
    # print nominator,denominator
    denominator = tf.cast(denominator, tf.float32)
    print nominator.eval() ,denominator.eval()
    print nominator, denominator
    sigma = tf.div(nominator, denominator)
    return sigma

def sigma(imgDis):
    length = imgDis.shape[0]
    denominator = np.divide(np.multiply(length, length-1), 2)
    nominator = np.divide(np.sum(imgDis), 2)
    sigma = np.divide(nominator, denominator)
    return sigma

def KDE(img_fs, sigma):
    return np.exp(- np.divide(img_fs, np.square(sigma) ) )

if __name__ == '__main__':
    with tf.Session() as sess:
        with tf.device("/gpu:0"):
            # img_fs = loadImgFeatureNumpy('data/vgg_feats.npy')
            img_fs_training = np.array([[1,2,3,4,5],[1,2,3,4,6],[1,2,3,4,6],[1,2,3,4,6],[1,2,3,4,7]], dtype = 'float32')
            # print 'vgg_feats.shape:', img_fs.shape
            # dis_all = imgDistanceMatrix(img_fs)
            # print 'dis_all.shape:', dis_all
            # np.save('imgDistance.npy', dis_all)
            img_fs_training = loadImgFeatureNumpy('data/vgg_feats_training.npy')
            # print 'vgg_feats_training.shape:', img_fs_training.shape
            # if img_fs_training.dtype != 'float64':
            #     print 'converting %s to float64'%img_fs_training.dtype
            #     img_fs_training.dtype = 'float64'
            print 'vgg_feats_training.shape:', img_fs_training.shape
            dis_training = imgDistanceTensor(img_fs_training)
            print 'dis_training.shape:', dis_training.get_shape()
            sigma_training = sigmaTensor(dis_training)
            print 'sigma:', sigma_training.eval()
            np.save('data/sigma_training.npy', sigma_training.eval())
            # kde_all = KDE(img_fs, sigma_training)
            # print 'KDE Matrix Shape:', kde_all.shape
            # np.save('kde_matrix.npy', kde_all)
