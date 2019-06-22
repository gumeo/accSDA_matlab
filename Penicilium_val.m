clear; clc;
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% open and read data
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
load penicilliumYES
Xa = X;
Ya = Y;

% the test samples
Io = [3,6,9,12];
Iout = [Io,Io+12,Io+24];

% the training samples:
Itr = setdiff(1:36,Iout);

X = Xa(Itr,:);
Y = Ya(Itr,:);
K = 3;
[n, p] = size(X)

Xtest = Xa(Iout,:);
Ytest = Ya(Iout,:);
[nt, pt] = size(Xtest)



% arrange data as columns of a matrix.  
% the first entry (top row) contains label 1 or 2


%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Extract training data matrix X.
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% First column is class labels; rest are observations.


% Center/normalize X.
% Center the data.
mu = mean(X)';
%muz = size(mu);
X = X - ones(n,1)* mu';

% Scale the data so each feature has variance equal to 1.

% Compute standard deviation of each feature.
sig = std(X);
In = find(sig~=0);

% Divide each feature by its standard deviation to ensure each feature has variance equal to one.
X = X(:,In) *diag(1./sig(In));
[n,p] = size(X);


%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Form indicator matrix Y.
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

% get number of classes (as largest class label).



%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Centroid matrix of training data.
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C = diag(1./diag(Y'*Y))*Y'*X;

%%
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Read and process test data.
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Xtest = Xtest - ones(nt,1)*mu';
Xtest = Xtest(:,In)*diag(1./sig(In));
[nt, pt] = size(Xtest)

% Center/normalize Xtest.




%% Set training, validation, and test sets.

% training.
train.X = X;
train.Y = Y;

% validation.
nv = 4; % 50 validation samples.
Ival = [1,5,9];
val.X = Xtest(Ival,:);
val.Y = Ytest(Ival,:);

% test.
Itst = setdiff(1:12,Ival);
Xtest = Xtest(Itst, :);
Ytest = Ytest(Itst, :);
[nt,~] = size(Xtest);

%% Estimate l1-penalty parameters.


%% SDAP.

% Set parameters.
%Om = eye(p) + 0.01*ones(p);
Om = eye(p);
gam = 0.001;
lams = [0.000001; 0.001; 0.005; 0.015; 0.05; 0.15; 0.5; 5];
q = 1;
PGsteps = 1000;
PGtol = 1e-5;
maxits = 250;
tol = 1e-3;
feat = 0.15;


% Call solver
display('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
display('SDAP = SDA + Proximal Gradient')
display('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
TPG2 = tic;
[B, ~, best_ind] = SDAP_val(train, val, Om, gam, lams, q, PGsteps, PGtol, maxits, tol, feat);
T = toc (TPG2);


% Test prediction.
B1 = B(:,:, best_ind);

% Project test data.
PXtest = Xtest*B1;
% Project centroids.
PC = C*B1;

% Compute distances to the centroid for each projected test observation.
dist = zeros(nt, K);
for i = 1:nt
    for j = 1:K
        dist(i,j) = norm(PXtest(i,:) - PC(j,:));
    end
end
  
% Label test observation according to the closest centroid to its projection.
[~,predicted_labels] = min(dist, [], 2);

% Form predicted Y.
Ypred = zeros(nt,K);
for i=1:nt
    Ypred(i, predicted_labels(i)) = 1;
end

% Number of misclassified.
fprintf('norm(B) %3.3e\nnnz(B) %g\nTime %g \n', norm(B1), nnz(B1)/p/q, T)
fprintf('# MC = %g\n', 1/2*norm(Ytest - Ypred, 'fro')^2)
fprintf('percent MC = %g\n\n', 1/2*norm(Ytest - Ypred, 'fro')^2/nt)


%% SDAAP.

% Set parameters.
%Om = eye(p) + 0.01*ones(p);
Om = eye(p);
gam = 0.001;
lams = [0.000001; 0.001; 0.005; 0.015; 0.05; 0.15; 0.5; 5];
q = 1;
PGsteps = 1000;
PGtol = 1e-5;
maxits = 250;
tol = 1e-3;
feat = 0.15;


% Call solver
display('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
display('SDAAP = SDA + Accelerated Proximal Gradient')
display('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
TAPG2 = tic;
[B, ~, best_ind] = SDAAP_val(train, val, Om, gam, lams, q, PGsteps, PGtol, maxits, tol, feat);
T3 = toc (TAPG2);


% Test prediction.
B3 = B(:,:, best_ind);

% Project test data.
PXtest = Xtest*B3;
% Project centroids.
PC = C*B3;

% Compute distances to the centroid for each projected test observation.
dist = zeros(nt, K);
for i = 1:nt
    for j = 1:K
        dist(i,j) = norm(PXtest(i,:) - PC(j,:));
    end
end
  
% Label test observation according to the closest centroid to its projection.
[~,predicted_labels] = min(dist, [], 2);

% Form predicted Y.
Ypred = zeros(nt,K);
for i=1:nt
    Ypred(i, predicted_labels(i)) = 1;
end

% Number of misclassified.
fprintf('norm(B) %3.3e\nnnz(B) %g\nTime %g \n', norm(B3), nnz(B3)/p/q, T3)
fprintf('# MC = %g\n', 1/2*norm(Ytest - Ypred, 'fro')^2)
fprintf('percent MC = %g\n\n', 1/2*norm(Ytest - Ypred, 'fro')^2/nt)



%% Call SDADMOM (ADMM).

% Set parameters.
%Om = eye(p) + gam*ones(p);
Om = eye(p);
gam = 0.001;
lams = [0.000001; 0.001; 0.005; 0.015; 0.05; 0.15; 0.5; 5];
mu = 1;
q = 1;
 PGsteps = 1000;
Pt.abs = 1e-5;
Pt.rel = 1e-5;
maxits = 250;
tol = 1e-3;
%t0 = 10*randn(K, 1); t0 =t0/norm(t0);


% Call solver
display('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
display('SDAD = SDA + ADMM')
display('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
TSMW = tic;
[B, Q, best_ind] = SDAD_val(train, val, Om, gam, lams, mu, q, PGsteps, Pt, maxits, tol, feat);
T2 =toc(TSMW);

% Extract solution.
B2 = B(:,:, best_ind);

% plot(B); hold('on')
% plot(B2, 'r');
% hold('off')
% fprintf('norm(B) %3.3e\nnnz(B) %g\nTime %g\n', norm(B2), nnz(B2), T2)


% Test prediction.

% Project test data.
PXtest = Xtest*B2;
% Project centroids.
PC = C*B2;

% Compute distances to the centroid for each projected test observation.
dist = zeros(nt, K);
for i = 1:nt
    for j = 1:K
        dist(i,j) = norm(PXtest(i,:) - PC(j,:));
    end
end
  
% Label test observation according to the closest centroid to its projection.
[~,predicted_labels] = min(dist, [], 2);

% Form predicted Y.
Ypred = zeros(nt,K);
for i=1:nt
    Ypred(i, predicted_labels(i)) = 1;
end

% Number of misclassified.
fprintf('norm(B) %3.3e\nnnz(B) %g\nTime %g \n', norm(B2), nnz(B2)/p/q, T2)
fprintf('# MC = %g\n', 1/2*norm(Ytest - Ypred, 'fro')^2)
fprintf('percent MC = %g\n\n', 1/2*norm(Ytest - Ypred, 'fro')^2/nt)

%% Plots.

plot(B1/norm(B1),'b'); hold('on')
plot(B2/norm(B2), 'r--');
plot(B3/norm(B3), 'g:');
legend('PG', 'ADMM', 'APG')
hold('off')


