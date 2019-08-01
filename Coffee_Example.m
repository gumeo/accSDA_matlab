% Coffee spectrogram classification example from Github README.

% Load data set.
clear; clc;
load('Data\Coffee.mat')


%% 
% Define required parameters.
p = 286; % Predictor variable sizes.
Om = eye(p); % Use identity matrix for Om.
gam = 1e-3; 
q = 1; % One fewer than number of classes.
quiet = true; % Suppress output.

% Optimization parameters.
insteps = 1000;
intol = 1e-4;
outsteps = 25;
outtol = 1e-4;

% Choose method and additional required arguments.
method = "APG"; % Use accelerated proximal gradient.
opts.bt = true; % Use backtracking.
opts.L = 0.25; % initial Lipschitz constant for BT.
opts.eta = 1.25; % scaling factor for BT.

% Use cross validation, and provide required arguments.
cv = true;
opts.folds = 7; % Use 7-fold CV as in paper.
opts.feat = 0.15; % Want classifiers using 15% of features.
lam = [1e-4; 1e-3; 1e-2; 1e-1]; % Choose lambda from this set.

%% Call ASDA.
ASDAres = ASDA(X, Y, Om, gam, lam, cv, method, q, insteps, outsteps, intol, outtol, quiet, opts);

%% Make plot.
plot(ASDAres.B)

%% Test predictions.

% First calculate centroid matrix of training data.
C = diag(1./diag(Y'*Y))*Y'*X;

% Call predict to calculate accuracy on testing data.
stats = predict(ASDAres.B, test, C')