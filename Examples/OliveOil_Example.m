% Coffee spectrogram classification example from Github README.

% Load data set.
clear; clc;
load('Data\OliveOil.mat')


%% 
% Define required parameters.
p = 570; % Predictor variable sizes.
Om = eye(p); % Use identity matrix for Om.
gam = 1e-3; 
q = 3; % One fewer than number of classes.
quiet = true; % Suppress output.

% Optimization parameters.
insteps = 1000;
intol = 1e-4;
outsteps = 25;
outtol = 1e-4;

% Choose method and additional required arguments.
method = "ADMM"; % Use alternating direction method of multiplier.
opts.mu = 5; % Set augmented Lagrangian parameter.

% Don't use cross validation. Set lambda.
cv = false;
lam = 1e-2; % Choose lambda.

%% Call ASDA.
ASDAres = ASDA(X, Y, Om, gam, lam, cv, method, q, insteps, outsteps, intol, outtol, quiet, opts);

%% Make plot.
plot(ASDAres.B)

%% Test predictions.

% First calculate centroid matrix of training data.
C = diag(1./diag(Y'*Y))*Y'*X;

% Call predict to calculate accuracy on testing data.
stats = predict(ASDAres.B, test, C')