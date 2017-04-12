# -*- encoding:utf8 -*-
'''
author: Jay Smelly
date: 2017.03.30
Reference: paper Reference Based LSTM for Image Captioning 
    ∗ weighted training part
version 2
'''
from nltk.stem import WordNetLemmatizer
try:
    import cPickle as pickle
except:
    import pickle 
import numpy as np
from tqdm import tqdm
import loadData

'''
Input : 
a array of words s1, s2, s3
'''
def normalized(words):
    denominator = np.sum(words)
    if denominator == 0:
        new_words = np.array([0]*len(words)) 
    else: 
        new_words = np.array([np.divide(s, denominator) for s in words])
        # print '2:',new_words.shape
    return new_words

def weighted_training():
    img_fs = loadData.loadNPY('data/vgg_feats_training.npy')  # vgg feats
    wordDict = loadData.loadPKL('data/wordDict.pkl') # word -> imgid
    trainset = loadData.loadJSON('data/trainset.json')
    sigma = loadData.loadMAT('data/sigma_training.mat')
    kde_matrix = loadData.loadMAT('data/kdeMatrix.mat')
    imgid2num = loadData.loadPKL('data/imgid2num.pkl') #imgid(string type) -> index of vgg_feats_training

    lengthOfImgfs = len(img_fs)
    import gc
    del img_fs
    gc.collect()
    caption_weights = []
    maxlen = 0

    print 'lengthOfTrainset:',len(trainset)
    print 'lengthOfImgfs:',lengthOfImgfs

    print 'computing weights...'
    for imgid in tqdm(trainset):
        for cap in trainset[imgid][0]['sentences']:
            gsi = set()
            maxlen = len(cap) if len(cap) > maxlen else maxlen
            new_l = []
            for word in cap:
                gsi = wordDict[word]
                assert(len(gsi)>0)
                wi = np.sum([kde_matrix[imgid2num[imgid]][imgid2num[imgjd]] for imgjd in gsi ])
                wi = np.divide(wi, len(gsi))
                new_l.append(wi)

            caption_weights.append(normalized(new_l))

    print 'transform to numpy array'
    zeros = np.zeros([len(captions[:100]), maxlen])
    for i,c in enumerate(caption_weights):
        zeros[i] += np.hstack((c, np.array((maxlen - len(c))*[0])))
    print 'saving at output/weights_ref.npy', 
    np.save('output/weights_ref.npy', zeros)

if __name__ == '__main__':
    weighted_training()
