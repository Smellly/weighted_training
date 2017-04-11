# simple binary tree
# in this implementation, a node is inserted between an existing node and the root

import sys

class BinaryTree():

    def __init__(self,rootid):
      self.left = None
      self.right = None
      self.rootid = rootid

    def getLeftChild(self):
        return self.left
    def getRightChild(self):
        return self.right
    def setNodeValue(self,value):
        self.rootid = value
    def getNodeValue(self):
        return self.rootid

    def insertRight(self,newNode):
        if self.right == None:
            self.right = BinaryTree(newNode)
        else:
            tree = BinaryTree(newNode)
            tree.right = self.right
            self.right = tree

    def insertLeft(self,newNode):
        if self.left == None:
            self.left = BinaryTree(newNode)
        else:
            tree = BinaryTree(newNode)
            self.left = tree
            tree.left = self.left


def printTree(tree):
        if tree != None:
            # sys.stdout.write('<')
            tmp = ''
            tmp += printTree(tree.getLeftChild())
            #print(tree.getNodeValue())
            tmp += ' ' + tree.getNodeValue()
            tmp += ' ' + printTree(tree.getRightChild())
            #sys.stdout.write('>')
            return tmp
        else:
            return ''

def getAllLeaves(tree):
    leaves = set()
    if tree != None:
        if tree.getLeftChild() is None and tree.getRightChild() is None:
            leaves.add(tree.getNodeValue())
        else:
            if tree.getLeftChild() is not None:
                leaves = leaves.union(getAllLeaves(tree.getLeftChild()))
            if tree.getRightChild() is not None:
                leaves = leaves.union(getAllLeaves(tree.getRightChild()))
        return leaves
    # else:
    #     return None

