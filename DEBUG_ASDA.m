% % Test on iris data set
% load fisheriris;
% X = normalize(meas);
% 
% % generate Yt
% Y = zeros(150,3);
% Y(1:50,1) = ones(50,1);
% Y(51:100,2) = ones(50,1);
% Y(101:150,3) = ones(50,1);
% 
% Om = eye(4);
% gam = 0.001;

% Test on ECG.
load('ECG.mat')


%%
Om = eye(136);
gam = 1e-3;
lam = [1e-4; 1e-3; 1e-2; 1e-1];

cv = true;

method = "APG";
opts.q = 1;
opts.insteps = 1500;
opts.outsteps = 10;
opts.intol = 1e-7;
opts.outtol = 1e-4;
opts.quiet = false;
opts.bt = true;
opts.L = 0.25;
opts.eta = 1.25;
opts.mu = 4;
opts.feat = 0.25;
opts.folds = 5;

%% Calculate DVs. 
[B,Q] = ASDA(Xt,Yt, Om, gam, lam, cv, method, opts);
plot(B);

%% 
scatterplot(Xt*B)