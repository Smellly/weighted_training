# -*- encoding:utf8 -*-
'''
author: Jay Smelly
date: 2017.03.30
Reference: paper Reference Based LSTM for Image Captioning 
    ∗ weighted training part
    * structured tree distance use bleu3
version 2
'''
import numpy as np
from nltk.translate.bleu_score import sentence_bleu as bleu
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
import loadData
import BinaryTree # printTree(t)

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

# compute cosine distance
def cosSimilar(inA,inB):  
    inA = np.mat(inA)  
    inB = np.mat(inB)  
    num = float(inA * inB.T)  
    denom = np.linalg.norm(inA) * np.linalg.norm(inB)  
    return 0.5 + 0.5 * (num / denom) 

def weighted_training():
    img_fs = loadData.loadNPY('data/vgg_feats.npy')  # vgg feats
    wordDict = loadData.loadPKL('data/wordDict.pkl') # word -> imgid
    trainset = loadData.loadJSON('data/trainset.json')
    sigma = loadData.loadMAT('data/sigma_training.mat')
    kde_matrix = loadData.loadMAT('data/kdeMatrix.mat')
    bow_array = loadData.loadNPY('data/bowArray.npy')
    imgid2num = loadData.loadPKL('data/imgid2num.pkl')
    pT = False
    if pT:
        import time

    lengthOfImgfs = len(img_fs)
    import gc
    del lengthOfImgfs
    gc.collect()
    caption_weights = []
    maxlen = 0

    print 'lengthOfTrainset:',len(trainset)
    print 'lengthOfImgfs:',lengthOfImgfs

    print 'computing weights...'
    for imgid in tqdm(trainset):
        if pT:
            t0 = time.time()
        for cap in trainset[imgid][0]['sentences']:
            if pT:
                t1 = time.time()
            gsi = set()
            maxlen = len(cap) if len(cap) > maxlen else maxlen
            new_l = []
            for word in cap:          
                if pT:
                    t2 = time.time()
                gsi = wordDict[word]
                assert(len(gsi) > 0)
                # compute p(I|w)
                wi = np.sum(
			[kde_matrix[imgid2num[imgid]][imgid2num[imgjd]] 
				for imgjd in gsi ])
                wi = np.divide(wi, len(gsi))
                # compute p(T|w)
                # the average bleu3 scores of 
                # two images'corresponding five structuredTrees 
                if pT:
                    t3 = time.time()
                    print 'wi time:', t3 - t2
                # compute cos distance at the same time
                wt = np.sum([
                    cosSimilar(
                        bow_array[imgid2num[imgid]], 
                        bow_array[imgid2num[imgjd]])
                            for imgjd in gsi])
                '''
                # read from kdeMatrix
                wt = np.sum([
                        cosDistMatrix[imgid2num[imgid]][imgid2num[imgjd]]
                            for imgjd in gsi])
                '''
                wt = np.divide(wt, len(gsi))
                if pT:
                    t4 = time.time()
                    print 'wt time:', t4 - t3
                w = np.multiply(wi, wt)
                new_l.append(w)
                if pT:
                    t5 = time.time()
                    print 'w time:', t5 - t4

            caption_weights.append(normalized(new_l))
            if pT:
                t6 = time.time()
                print 'sentence time:', t6 - t1
        if pT:
            print 'imgid time:', time.time() - t0
    print 'transform to numpy array'
    zeros = np.zeros([len(captions), maxlen])
    for i,c in enumerate(caption_weights):
        zeros[i] += np.hstack((c, np.array((maxlen - len(c))*[0])))
    print 'saving at output/weights_ref.npy', 
    np.save('output/weights_ref.npy', zeros)

if __name__ == '__main__':
    '''
    a = np.array([1,2,3])
    b = np.array([1,2,3])
    c = np.array([2,2,3])
    print cosSimilar(a,b)
    print cosSimilar(a,c)
    '''
    weighted_training()
