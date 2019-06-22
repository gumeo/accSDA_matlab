function [stats, preds] = test_ZVD(w, test, classMeans, mus, scaling)
% Classify test data using nearest centroid classification and 
% discriminant vectors learned from the training set.  
%====================================================================
% Input.
%====================================================================
% w = matrix with columns equal to discriminant vectors.
% test = matrix containing test set
% classMeans = means of each class in the training set 
%	(used for computing centroids for classification)
% mus = means/standard devs of the training set.
%	(used for centering/normalizing the test data appropriately)
%====================================================================
% Output. 
%====================================================================
% stats: list containing #misclassified observations, l0 and 
%			l1 norms of discriminants.
% pred: predicted class labels according to nearest centroid and the 
%			discriminants.


%====================================================================
% Initialization.
%====================================================================
  
% Get scaling/centering factors.
if (scaling==1)
    mu = mus.mu;
    sig = mus.sig;
else
    mu = mus;
end % if.

% Extract class labels and observations.
test_labels = test(:,1);
test_obs = test(:,2:size(test,2));

% Get number of test observations.
N = length(test_labels);

% Get number of classes.
K = max(test_labels);

% Center the test data.
test_obs = test_obs - ones(N,1)*mu';

% Scale according to the saved scaling from the training data (if desired)
if (scaling==1)
    test_obs = test_obs*diag(1./sig);
end

%====================================================================
% Classify the test data according to nearest centroid rule.
%====================================================================

% Project the test data to the lower dim linear space defined by the ZVDs.
proj = w'*test_obs';
   
% Compute centroids and projected distances.
cent = w'*classMeans;
  
% Compute distances to the centroid for each projected test observation.
dist = zeros(N, K);
for i = 1:N
    for j = 1:K
        dist(i,j) = norm(proj(:,i) - cent(:,j));
    end
end
  
% Label test observation according to the closest centroid to its projection.
[~,predicted_labels] = min(dist, [], 2);
  
%===================================================================
% Compute fraction misclassed, l0 and l1 norms of the classifiers.
%====================================================================
  
% Compute fraction of misclassified observations.
misclassed = sum(abs(test_labels - predicted_labels) > 0) / N;

% l0
l0 = sum(abs(w)>1e-3);

% l1
l1 = sum(abs(w));
  
  
%====================================================================
% Output results.
%====================================================================

% stats.
stats.mc = misclassed;
stats.l0 =l0;
stats.l1 = l1;

%labels.
preds = predicted_labels;

 