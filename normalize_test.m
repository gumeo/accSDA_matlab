function x=normalize_test(x, mu, sig, feat)
% NORMALIZE_TEST Transforms test data x according to training mean and
% variance.
% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% INPUT.
% x - testing data.
% mu - sample mean of training data.
% sig - sample standard deviation of training data
% feat - vector of indices of nonconstant training features.
% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% OUTPUT.
% x - transformed testing data.
% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

n = size(x,1);

% Delete nontrivial features.
x = x(:, feat);

% Shift and scale by mu and sigma.
x=x-ones(n,1)*mu;
x=x*diag(1./sig);
end