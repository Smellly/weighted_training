from tqdm import tqdm
import scipy.io as scio
import pickle
import numpy as np
import BinaryTree
import loadData

def collectLeaves():
    ind = loadData.loadJSON('data/cleanTree/mTreeNodeIndex.json')['train']
    with open('data/cleanTree/cleanEntityList_noChange_removeV.txt' ,'r') as f:
        dlist = f.readlines()
    print 'len of ind:', len(ind)
    wordSet = set()
    for t in tqdm(ind):
        for l in ind[t]:
            for i,num in enumerate(l):
                if i%2 != 0:
                    continue
                wordSet.add(dlist[num-1].split('#')[0])
    wordList = sorted(list(wordSet))
    print 'length of wordSet:', len(wordList)
    with open('data/wordSet4bow.pkl', 'w') as f:
        pickle.dump(wordList, f, True)
    return wordList

# matrix is followed by cooc filename order
def buildBow(wordList):
    ind = loadData.loadJSON('data/cleanTree/mTreeNodeIndex.json')['train']
    with open('data/cleanTree/cleanEntityList_noChange_removeV.txt' ,'r') as f:
        dlist = f.readlines()

    rows = len(ind)
    columns = len(wordList)
    bow = np.zeros([rows,columns])

    for i,t in enumerate(tqdm(ind)):
        for l in ind[t]:
            for j,num in enumerate(l):
                if j%2 != 0:
                    continue
                bow[i][wordList.index(dlist[num-1].split('#')[0])] = 1
    np.save('data/cleanBowArray.npy', bow) 
    scio.savemat('data/cleanBowMatrix.mat', {'bow': bow})


def test():
    path = 'data/structuredTree/results_coco/parserTree_train2014_r1.txt'
    tree = loadData.loadPKL(path)
    w = set()
    for i,t in enumerate(tree):
        print BinaryTree.printTree(t)
        print BinaryTree.getAllLeaves(t)
        w = w.union(BinaryTree.getAllLeaves(t))
        if i == 10:
            break
    if None in w:
        print 'removing None'
        w.remove(None)
    print len(w)

def collectLeavesPKL():
    wordSet = set()
    path_fore = 'data/structuredTree/results_coco/parserTree_train2014_r'
    for i in range(1,6):
        path = path_fore + str(i) + '.txt'
        tree = loadData.loadPKL(path)
        for t in tqdm(tree):
            wordSet = wordSet.union(BinaryTree.getAllLeaves(t))
    if None in wordSet:
        print 'removing None'
        wordSet.remove(None)
    print 'length of wordSet:', len(wordSet)
    with open('data/cleanWordSet4bow.pkl', 'w') as f:
        pickle.dump(wordSet, f, True)
    
def buildBowMatrixPKL(wordList):
    # path_ws = 'data/wordSet4bow.pkl'
    # wordSet = loadData.loadPKL(path_ws)
    # wordList = sorted(list(wordSet))
    columns = len(wordList)
    path_tree = 'data/structuredTree/results_coco/parserTree_train2014_r1.txt'
    tree = loadData.loadPKL(path_tree)
    rows = len(tree)
    bow = np.zeros([rows,columns])
    path_fore = 'data/structuredTree/results_coco/parserTree_train2014_r'
    for j in range(1,6):
        path = path_fore + str(j) + '.txt'
        tree = loadData.loadPKL(path)
        for i,t in enumerate(tqdm(tree)):
            w = set()
            w = w.union(BinaryTree.getAllLeaves(t))
            for leaf in w:
                bow[i][wordList.index(leaf)] = 1
        np.save('bowArray'+str(j)+'.npy', bow)
    np.save('bowArray.npy', bow) 

if __name__ == '__main__':
    # test()
    wordList = collectLeaves()
    buildBow(wordList)
