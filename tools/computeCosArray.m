
%
addpath('../../npy-matlab');
mFeats = readNPY('../data/bowArray.npy');
size(mFeats)
bowCosA = pdist(mFeats, 'cosine');
clear mFeats
size(bowCosA)
eval(['save -v7.3 ../data/bowCosArray.mat bowCosA']);
bowCosM = squareform(bowCosA);
clear bowCosA
eval(['save -v7.3 ../data/bowCosMatrix.mat bowCosM']);
size(bowCosM)
%}



