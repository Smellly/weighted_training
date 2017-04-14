
%
mFeats = load('../data/cleanBowMatrix.mat');
mFeats2 = mFeats.bow;
clear mFeats
size(mFeats2)
mDist = pdist2(mFeats2, mFeats2, 'cosine');
eval(['save -v7.3 ../data/cleanDistanceMatrix.mat mDist']);
size(mDist)
%save('mDistanceMatrix.mat','mDist')
%}



