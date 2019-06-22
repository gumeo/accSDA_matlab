clear all;clc;

% Test on iris data set
load fisheriris;
Xt = meas;

% generate Yt
Yt = zeros(150,3);
Yt(1:50,1) = ones(50,1);
Yt(51:100,2) = ones(50,1);
Yt(101:150,3) = ones(50,1);

Om = eye(4);
gam = 0.001;

% Some value, maybe change it
train.X = Xt;
train.Y = Yt;
folds=2;
gams = [0.01, 0.02];
beta = 0.05;
D = eye(4);
q = 2;
maxits = 10; % Just a low number to test
tol = 1e-3;
ztol = 1e-3;
feat = 0.8;
quiet = 0;

[B,Q] = SZVDcv(train, folds, gams, beta, D, q, maxits, tol, ztol, feat, quiet);

