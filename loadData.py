# -*- encoding:utf8 -*-
'''
author: Jay Smelly
date: 2017.03.30

load data
'''

try:
    import cPickle as pickle
except:
    import pickle
import scipy.io as scio  
import h5py
import json
import numpy as np
import BinaryTree # printTree(t)
from nltk.stem import WordNetLemmatizer

def loadMAT(path, p = True):
    if p:
        print 'loading', path
    try:
        mat = scio.loadmat(path)
    except NotImplementedError:
        mat = h5py.File(path)
    # print 'mat keys:', mat.keys()
    name = mat.keys()[0]
    # print mat[name].shape
    return np.transpose(mat[name])

def loadNPY(path, p = True):
    if p:
        print 'loading', path
    return np.load(path)

def loadPKL(path, p = True):
    if p:
        print 'loading', path
    with open(path, 'r') as f:
        pkl = pickle.load(f)
    return pkl

def loadJSON(path, p = True):
    if p:
        print 'loading', path
    with open(path, 'r') as f:
        js = json.load(f)
    return js
    
def loadImgFeatureNumpy(path, p = True):
    if p:
        print 'loading', path
    img_fs = np.load(path)
    return img_fs

def loadTrainSet(dataset_path):
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    return dataset

def loadCaptions(dataset_path):
    lemmatizer = WordNetLemmatizer()
    dataset = json.load(open(dataset_path, 'r'))
    # imgid is vgg_feats column
    raw = dataset['images'] # 123287 * 5 
    captions = []
    for i in range(5):
        part = []
        for s in raw:
            new_s = [lemmatizer.lemmatize(word) 
                for word in s['sentences'][i]['tokens']]
            part.append(new_s)
        captions.append(part)

    return captions

def loadStructuredTrees(
        path='resourse/structuredTree/results_coco/parserTree_train2014_r'):
    trees = []
    # read trees
    for i in range(1,6):
        treepath = path + str(i) + '.txt'
        with open(treepath, 'r') as f:
            tmp = pickle.load(f)
        trees.append(tmp)
    # read coco_train.txt
    with open(
        'resourse/structuredTree/coco_train.txt', 'r') as f:
        jpgname = f.readlines()
    # print len(trees) # 413915 is five times the training set

    treedict = [ dict(zip(jpgname, trees[i])) for i in range(5)]

    dataset = json.load(open('dataset.json', 'r'))
    jpgnamelist = [dataset['images'][i]['filename'] 
                        for i in range(len(dataset['images']))]
    imageidlist = [dataset['images'][i]['imgid']    
                        for i in range(len(dataset['images']))]
    # print len(jpgnamelist),len(imageidlist) 123287 is the all dataset including training val and test
    assert(len(jpgnamelist) == len(imageidlist))
    jpgnamedict = dict(zip(jpgnamelist, imageidlist))
    
    imgidtree = []
    for d in treedict:
        new_dict = dict()
        for k in d:
            imgid = jpgnamedict[k.strip()]
            new_dict[imgid] = d[k]
        imgidtree.append(sorted(new_dict.items(), key = lambda d:d[0]))

    # tmp2 = sorted(tmp.items(), key=lambda d: d[1], reverse = False) # 413915
    # t = [BinaryTree.printTree(i[0]).split() for i in tmp2]
    return imgidtree
'''
if __name__ == '__main__':
    # construct()
    d = constructTrainSet('data/dataset.json')
    with open('data/trainset.json', 'w') as f:
        json.dump(d, f)
    for i in d:
        print i,d[i]
        break
'''
