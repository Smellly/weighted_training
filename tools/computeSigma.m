DistArrayStruct = load('data/mDistanceMatrix.mat');
DistArray = DistArrayStruct.mDist;
denominator = length(DistArray)
nominator = sum(DistArray)
sigma = nominator/denominator
eval(['save -v7.3 sigma_training.mat sigma']);
