# -*- encoding:utf8 -*-
'''
author: Jay Smelly
date: 2017.03.30
Reference: paper Reference Based LSTM for Image Captioning 
    ∗ weighted training part
construct the training set
储存说明：
    dictionary:
    img_id:
            captions:cap1~cap5
            trees:t1~t5

version: 3
'''

try:
    import cPickle as pickle
except:
    import pickle
import scipy.io as scio  
import json
import numpy as np
import BinaryTree # printTree(t)
from nltk.stem import WordNetLemmatizer
import loadData
from tqdm import tqdm
import os

def construct():
    vgg_feats_raw = loadImgFeatureNumpy('vgg_feats.npy')
    captionss_raw = loadCaptions('dataset.json')                                                                    
    d = loadStructuredTrees()
    print 'd length:',len(d)
    # redundent_imgid = [t[1] for t in d]
    imgid = []
    tmp = -1
    for t in d[0]:
        if t[0] != tmp:
            tmp = t[0]
            imgid.append(t[0])
    print 'length of imgid:',len(imgid)
    vgg_feats_training = [vgg_feats_raw[i] for i in imgid]
    # captions_training  = [captionss_raw[i] for i in redundent_imgid] # bug
    captions_training = []
    for caps in captionss_raw:
        tmp = []
        for i in imgid:
            tmp.append(caps[i])
        captions_training.append(tmp)
    vgg_feats_training = np.array(vgg_feats_training)
    structuredTree_training = []
    for t in d:
        tmp = [BinaryTree.printTree(i[1]).split() for i in t]
        structuredTree_training.append(tmp)

    print 'imgid:',imgid[:10]
    print vgg_feats_training.shape
    print 'length of captions_training',len(captions_training)
    np.save('vgg_feats_training.npy', vgg_feats_training)
    with open('captions_training.pkl', 'w') as fout:
        pickle.dump(captions_training, fout, True)
    
    with open('structuredTree_training.pkl', 'w') as fout:
        pickle.dump(structuredTree_training, fout, True)
    print 'sTree:',structuredTree_training[0][:2]
    print 'captions:',captions_training[0][:2]
    print 'sTree:',structuredTree_training[4][-2:]
    print 'captions:',captions_training[4][-2:]

def constructTrainSet(dataset_path):
    lemmatizer = WordNetLemmatizer()
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    trainDict = dict()
    for item in dataset['images']:
        if item['split'] == 'train':
            sents = []
            for cap in item['sentences']:
                new_s = [lemmatizer.lemmatize(word)
                    for word in cap['tokens']]
                sents.append(new_s)
            trainDict[item['imgid']] = [
                dict(sentences = sents),
                dict(trees = []),
                dict(filename = item['filename'])]
    return trainDict

def constructWordDict():
    trainset = loadData.loadTrainSet('data/trainset.json')
    wordset_path = 'data/wordset.pkl'
    if os.path.isfile(wordset_path):
        print 'loading wordset'
        with open(wordset_path, 'r') as f:
            wordset = pickle.load(f)
    else:
        wordset = set() 
        print 'constructing wordset'
        for imgid in tqdm(trainset):
            for cap in trainset[imgid][0]['sentences']:
                wordset = wordset.union(cap)
        print 'saving at', wordset_path
        with open(wordset_path, 'w') as f:
            pickle.dump(wordset, f, True)

    print '# of words:', len(wordset)

    wordDict = dict().fromkeys(wordset, None)
    print 'constructing wordDict: appending imgid'
    i = 0
    for word in tqdm(wordset):
        for imgid in trainset:
            for cap in trainset[imgid][0]['sentences']:
                if word in cap:
                    if wordDict[word] is None:
                        wordDict[word] = [imgid]
                    else:
                        wordDict[word].append(imgid)
                    break

    print 'constructing wordDict: removing reduntent items'
    for word in tqdm(wordDict):
        wordDict[word] = list(set(wordDict[word]))

    print 'for validation..'
    for i,word in enumerate(wordDict):
        print word, len(wordDict[word]), len(set(wordDict[word]))
        if len(word) < 20:
            print wordDict[word]
        if i == 10:
            break

    wordDict_path = 'data/wordDict.pkl'
    print 'saving wordDict at', wordDict_path
    with open(wordDict_path, 'w') as f:
        pickle.dump(wordDict, f, True)

def addTrees():
    trainset_path = 'data/trainset.json'
    if os.path.isfile(trainset_path):
        print 'loading trainset'
        with open(trainset_path, 'r') as f:
            trainset = json.load(f)
    else:
        print 'construct trainset'
        trainset = constructTrainSet('data/dataset.json')
        with open(trainset_path, 'w') as f:
            json.dump(trainset, f)
    # print type(trainset)

    # read coco_train.txt
    with open(
    '../2017.03.11ICCV/resourse/structuredTree/coco_train.txt', 'r') as f:
        jpgname = f.readlines()
        
    print 'constructing filename to image_id dictionary'
    dataset = json.load(open('../2017.03.11ICCV/dataset.json', 'r'))
    jpgnamelist = [dataset['images'][i]['filename'] 
                        for i in range(len(dataset['images']))]
    imageidlist = [dataset['images'][i]['imgid']    
                        for i in range(len(dataset['images']))]
    # print len(jpgnamelist),len(imageidlist) 
    # 123287 is the all dataset including training val and test
    assert(len(jpgnamelist) == len(imageidlist))
    jpgnamedict = dict(zip(jpgnamelist, imageidlist))

    tree_path = \
        '../2017.03.11ICCV/resourse/structuredTree/results_coco/parserTree_train2014_r'

    for i in range(1,6):
        treepath = tree_path + str(i) + '.txt'
        trees = []
        with open(treepath, 'r') as f:
            trees = pickle.load(f)
        assert(len(trees) == len(trainset)) # 82783

        print 'constructing %dth filename to tree dictionary'%i
        treedict = dict(zip(jpgname, trees))
        
        print 'constructing %dth image_id to tree dictionary'%i
        imgidDict = dict()
        for d in treedict:
            # d is jpgname(i.e. filename)       
            imgid = jpgnamedict[d.strip()]
            imgidDict[imgid] = treedict[d]
            # print imgid
            # print BinaryTree.printTree(treedict[d])

        for imgid in imgidDict:
            # try:
            trainset[str(imgid)][1]['trees'].append(
                BinaryTree.printTree(imgidDict[imgid]).split())
            # except:
            #     for j in sorted(trainset.keys(), key = lambda d:d[0], reverse = True):
            #         print imgid, j, type(j)

    savepath = 'data/trainset_t.json'
    print 'saving new trainset at ',savepath
    with open(savepath, 'w') as f:
        json.dump(trainset, f)



if __name__ == '__main__':
    # construct()
    # d = constructTrainSet('data/dataset.json')
    # with open('data/trainset.json', 'w') as f:
    #     json.dump(d, f)
    # print len(d)
    # for i in d:
    #     print i,d[i]
    #     break
    # constructWordDict()
    addTrees()
