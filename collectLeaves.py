from tqdm import tqdm
import pickle
import numpy as np
import BinaryTree
import loadData

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

def collectLeaves():
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
    with open('data/wordSet4bow.pkl', 'w') as f:
        pickle.dump(wordSet, f, True)
    
def buildBowMatrix():
    path_ws = 'data/wordSet4bow.pkl'
    wordSet = loadData.loadPKL(path_ws)
    wordList = sorted(list(wordSet))
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
    # collectLeaves()
    buildBowMatrix()
