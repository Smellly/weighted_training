import loadData
import numpy as np

def tri(x,y):
    if x<y:
        x,y = y,x
    length = 82783
    return length*(y) - y*(y+1)/2 + x-y-1

'''
m = np.random.rand(1,100)
n = np.reshape(m, [10,10]).T
# print m
# print n
print 4,0,tri(4,0), m[0][tri(4,0)], n[4][0]
print 4,1,tri(4,1), m[0][tri(4,1)], n[4][1]
print 4,2,tri(4,2), m[0][tri(4,2)], n[4][2]
print 3,7,tri(3,7), m[0][tri(3,7)], n[7][3]
print 1,4,tri(1,4), m[0][tri(1,4)], n[4][1]
print 7,6,tri(7,6), m[0][tri(7,6)], n[7][6]
'''

m = loadData.loadMAT('data/kdeArray.mat')
n = loadData.loadMAT('data/kdeMatrix.mat')

print 4,0,tri(4,0), m[0][tri(4,0)], n[4][0]
print 4,1,tri(4,1), m[0][tri(4,1)], n[4][1]
print 4,2,tri(4,2), m[0][tri(4,2)], n[4][2]
print 3,7,tri(3,7), m[0][tri(3,7)], n[7][3]
print 1,4,tri(1,4), m[0][tri(1,4)], n[4][1]
print 7,6,tri(7,6), m[0][tri(7,6)], n[7][6]
