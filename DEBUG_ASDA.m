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
load('ECGdata (normalized).mat')


%%
lam = 1e-2;

cv = false;

method = "APG";
opts.q = 2;
opts.insteps = 1500;
opts.outsteps = 10;
opts.intol = 1e-7;
opts.outtol = 1e-4;
opts.quiet = false;
opts.bt = false;
% opts.L = 0.25;
% opts.eta = 1.25;
opts.mu = 4;

%% Calculate DVs. 
[B,Q] = ASDA(X,Y, Om, gam, lam, cv, method, opts);
plot(B);

%% 
scatterplot(X*B)