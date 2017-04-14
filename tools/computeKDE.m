% DistArrayStruct = load('data/mDistanceMatrix.mat');
mFeats = load('../data/cleanBowMatrix.mat');
mFeats2 = mFeats.vgg_feats_training;
mDist = pdist(mFeats2);
eval(['save -v7.3 ../data/cleanDistanceMatrix.mat mDist']);
% DistArray = DistArrayStruct.mDist;
% clear DistArrayStruct
sigmaStruct = load('data/sigma_training.mat')
sigma = sigmaStruct.sigma
kdeArray = exp(-DistArray/sigma*sigma);
clear DistArray
eval(['save -v7.3 data/kdeArray.mat kdeArray']);
size(kdeArray)
kdeMatrix = squareform(kdeArray, 'tomatrix');
size(kdeMatrix)
eval(['save -v7.3 data/kdeMatrix.mat kdeMatrix']);

