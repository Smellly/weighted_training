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
import time
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
    img_fs = loadData.loadNPY('data/vgg_feats.npy')  # vgg feats
    wordDict = loadData.loadPKL('data/wordDict.pkl') # word -> imgid
    trainset = loadData.loadJSON('data/trainset_t.json')
    sigma = loadData.loadNPY('data/sigma_training_fasle.npy')
    kde_matrix = loadData.loadNPY('data/kde_matrix_false.npy')
    imgid2num = loadData.loadPKL('data/imgid2num.pkl')
    pT = True

    lengthOfImgfs = len(img_fs)
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
                wi = np.sum([kde_matrix[imgid2num[imgid]][imgid2num[imgjd]] for imgjd in gsi ])
                wi = np.divide(wi, len(gsi))
                # compute p(T|w)
                # the average bleu3 scores of 
                # two images'corresponding five structuredTrees 
                if pT:
                    t3 = time.time()
                    print 'wi time:', t3 - t2
                scores = 0
                for imgid_j in gsi:
                    '''
                    r = np.random.randint(0,5)
                    Ti = trainset[imgid][1]['trees'][r]
                    Tj = trainset[imgid_j][1]['trees'][r]
                    scores = bleu(
                            [Ti], Tj, weights = (0.3333, 0.3333, 0.3333))
                    '''
                    scores += np.mean([
                        bleu([Ti], Tj, weights = (0.3333, 0.3333, 0.3333))
                        for Ti,Tj in zip(
                            trainset[imgid][1]['trees'],
                            trainset[imgid_j][1]['trees']
                            )
                        ])
                    
                wt = np.divide(scores-5, len(gsi)) # bleu(Ti, Ti) = 1 five times
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
    weighted_training()
