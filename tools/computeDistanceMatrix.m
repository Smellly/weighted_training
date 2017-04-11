
%
mFeats = load('vgg_feats_training.mat');
mFeats2 = mFeats.vgg_feats_training;
mDist = pdist(mFeats2);
eval(['save -v7.3 mDistanceMatrix.mat mDist']);
%save('mDistanceMatrix.mat','mDist')
%}



