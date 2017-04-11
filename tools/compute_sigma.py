# -*- encoding:utf8 -*-
'''
author: Jay Smelly
date: 2017.03.12
Reference: paper Reference Based LSTM for Image Captioning ∗ weighted training part
compute sigma
'''

import numpy as np

def loadImgFeatureNumpy(path):
    img_fs = np.load(path)
    return img_fs

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

def sigma(imgDis):
    length = imgDis.shape[0]
    denominator = np.divide(np.multiply(length, length-1), 2)
    nominator = np.divide(np.sum(imgDis), 2)
    sigma = np.divide(nominator, denominator)
    return sigma

def KDE(img_fs, sigma):
    return np.exp(- np.divide(img_fs, np.square(sigma) ) )

if __name__ == '__main__':
    img_fs = loadImgFeatureNumpy('vgg_feats.npy')
    img_fs_training = loadImgFeatureNumpy('vgg_feats_training.npy')
    # img_fs_training = np.array([[1,2,3,4,5],[1,2,3,4,6],[1,2,3,4,6],[1,2,3,4,6],[1,2,3,4,7]])
    print 'vgg_feats.shape:', img_fs.shape
    print 'vgg_feats_training.shape:', img_fs_training.shape
    dis_all = imgDistanceMatrix(img_fs)
    print 'dis_all.shape:', dis_all
    np.save('imgDistance.npy', dis_all)
    dis_training = imgDistanceMatrix(img_fs_training)
    print 'dis_training.shape:', dis_training.shape
    sigma_training = sigma(dis_training)
    print 'sigma:', sigma_training
    np.save('sigma_training.npy', sigma_training)
    kde_all = KDE(img_fs, sigma_training)
    print 'KDE Matrix Shape:', kde_all.shape
    np.save('kde_matrix.npy', kde_all)
    