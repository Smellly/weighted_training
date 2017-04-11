DistArrayStruct = load('data/mDistanceMatrix.mat');
DistArray = DistArrayStruct.mDist;
clear DistArrayStruct
sigmaStruct = load('data/sigma_training.mat')
sigma = sigmaStruct.sigma
kdeArray = exp(-DistArray/sigma*sigma);
clear DistArray
eval(['save -v7.3 data/kdeArray.mat kdeArray']);
size(kdeArray)
kdeMatrix = squareform(kdeArray, 'tomatrix');
size(kdeMatrix)
eval(['save -v7.3 data/kdeMatrix.mat kdeMatrix']);

