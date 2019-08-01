function [stats,preds,proj,cent, dist]=predict(w,test,classMeans)
% PREDICT calculates cardinality and classification error.
% Calculates cardinality and classification error for given discriminant
% vectors "w", test data "test", and centroids "classMeans".
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% INPUT.
%w: matrix with columns are discriminant vectors
%test:test data; in form [labels, data].
%ClassMeans: means of each class in the training data: R'
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% OUTPUT.
% stats: list containing misclassified obs, l0 and l1 norms of discriminant
% preds: predicted labels for test data according to nearest centroid and
%   the discriminants

%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% COMPUTE PROJECTION OF TEST DATA.
% Extract class labels and obs
test_labels=test(:,1);
test_obs=test(:,2:end);

% Get number of test obs and classes.
N=size(test,1);
K=length(unique(test_labels));

% Project the test data onto column space of w.
proj=w'*test_obs';

% Compute the projected centroids.
cent=w'*classMeans;

%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% COMPUTE SCORES.

% Compute the distances to the centroid for test data, i.e the distance of
% every projected point to every cent:
dist=zeros(N,K);
for i=1:N
    for j=1:K
        dist(i,j)=norm(proj(:,i)-cent(:,j));
    end
end

% Label test_obs accoring to the closest centroid to its projection
[~,preds]=min(dist, [], 2);

% Compute fraction of misclassified observations
misclassed=sum(abs(test_labels-preds)>0)/N;

% Compute number of nonzero features (with abs value > 1e-3);
l0 = sum(abs(w)>1e-3); %l0 is the number of non-zero entries

% Compute l1 norm of w.
l1=sum(abs(w));

% Prepare output.
stats.mc=misclassed;
stats.l0=sum(l0);
stats.l1=l1;




    

