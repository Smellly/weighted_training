mDistA = load('data/mDistanceMatrix.mat')
mDistM = squareform(mDistA, 'tomatrix')
eval(['save -v7.3 data/mDistMatrix.mat mDistM'])
