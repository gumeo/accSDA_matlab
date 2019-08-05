function [x, mu, sig, feat] = normalize(x)
% NORMALIZE Transforms data x to have column mean 0 and variance 1.
% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% INPUT.
% x - data matrix with observations as rows.
% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% OUTPUT.
% x - standardized data matrix.
% mu - sample mean of x.
% sig - sample standard deviation of x.
% feat - vector of indices of nonconstant features.
% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

% Get dimension of data matrix.
% n = number of observations, p = number of features.
[n,p]=size(x);

% Calculate sample mean.
mu = mean(x);

% Shift x to have sample mean 0.
x=x-ones(n,1)*mu;

% Calculate standard deviation.
sig=std(x);

% Delete features with variance equal to 0.
const = false(p,1);
for i = 1:p
    if sig(i) == 0
        const(i) = true;
    end
end

% Get list of nonconstant features.
feat = 1:p;
feat(const) = [];

% Extract only entries of x, mu, sig corresponding to nontrivial features.
x = x(:, feat);
sig = sig(feat);
mu = mu(feat);

% Normalize x to have variance 1.
x=x*diag(1./sig);

end
