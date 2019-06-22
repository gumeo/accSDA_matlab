clear; clc;
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% open and read data
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
fid = fopen('ECGFiveDays_TRAIN','r');
A = fscanf(fid, '%g');
fclose(fid);

% arrange data as columns of a matrix.  
% the first entry (top row) contains label 1 or 2
B=reshape(A,137,23);  % change long data vectors into a matrix of size 137x23
B = B';

%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Extract training data matrix X.
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% First column is class labels; rest are observations.
[n,p] = size(B);
X = B(:, 2:p);
p = p-1;

% Center/normalize X.
% Center the data.
mu = mean(X)';
%muz = size(mu);
X = X - ones(n,1)* mu';

% Scale the data so each feature has variance equal to 1.

% Compute standard deviation of each feature.
sig = std(X);

% Divide each feature by its standard deviation to ensure each feature has variance equal to one.
X = X *diag(1./sig);

%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Form indicator matrix Y.
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

% get number of classes (as largest class label).
K = max(B(:,1));

% Initialize Y.
Y = zeros(n,K);

% Assign values according to label.
for i=1:n
    Y(i, B(i,1)) = 1; % Set Yij = 1 if label(i) = j.
end

%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Centroid matrix of training data.
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C = diag(1./diag(Y'*Y))*Y'*X;

%%
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Read and process test data.
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
fid=fopen('ECGFiveDays_TEST','r');
AT=fscanf(fid,'%g');
fclose(fid);
BT=reshape(AT,137,861);
Btest = BT';

% Make Xtest.
[nt, pt] = size(Btest);
Xtest = Btest(:, 2:pt);
pt = pt - 1;

Xtest = Xtest - ones(nt,1)*mu';
Xtest = Xtest*diag(1./sig);

% Center/normalize Xtest.


% Make Ytest.
Ytest = zeros(nt, K);
for i = 1:nt
    Ytest(i, Btest(i,1)) = 1;    
end



%% Set training, validation, and test sets.

% training.
train.X = X;
train.Y = Y;

% % validation.
% nv = 50; % 50 validation samples.
% val.X = Xtest(1:nv,:);
% val.Y = Ytest(1:nv,:);

% test.
[nt,~] = size(Xtest);


%% Estimate l1-penalty parameters.

% Set parameters.
%Om = eye(p) + 0.01*ones(p);
Om = eye(p);
gam = 0.001;


% Form A.
A = 2*(X'*X + gam*Om);

%++++++++++++++++++++++++++++++++++++
% Initialize theta.
%++++++++++++++++++++++++++++++++++++
% Precompute Mj = I - Qj*Qj'*D.
Qj = ones(K, 1);
D = 1/n*(Y'*Y);
Mj = @(u) u - Qj*(Qj'*(D*u));

% Initialize theta.
theta = Mj(rand(K,1));
theta = theta/sqrt(theta'*D*theta);

%%
% Form d.
d = 2*X'*Y*theta;

% Initialize beta.
beta = A\d; % 1st unpenalized solution.

% Choose lambda so that unpenalized solution always has negative value.
lmax = (beta'*d - 0.5*beta'*A*beta)/norm(beta, 1);


lams = 2.^(-9:1:3)*lmax;




%% SDAP.

%lams = [0.001; 0.005; 0.015; 0.05; 0.15; 0.5; 5];
q = 1;
quiet = 1;
PGsteps = 1000;
PGtol = 1e-5;
maxits = 250;
tol = 1e-3;
feat = 0.15;


folds = 5;

% Call solver
display('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
display('SDAP = SDA + Proximal Gradient')
display('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
TPG2 = tic;
[B, ~, ~] = SDAPcv(train, folds, Om, gam, lams, q, PGsteps, PGtol, maxits, tol, feat, quiet);
T = toc (TPG2);

%

% Test prediction.
B1 = B(:,:);
B1 = B1/norm(B1);

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
% Om = eye(p);
% gam = 0.001;
% lams = [0.001; 0.005; 0.015; 0.05; 0.15; 0.5; 5];
q = 1;
%quiet = 0;
PGsteps = 1000;
PGtol = 1e-5;
maxits = 250;
tol = 1e-3;
feat = 0.15;


folds = 5;

% Call solver
display('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
display('SDAAP = SDA + Accelerated Proximal Gradient')
display('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
TAPG2 = tic;
%[B, ~] = SDAAP_CV(train, folds, Om, gam, lams, q, PGsteps, PGtol, maxits, tol, feat)
[B2, ~, lbest] = SDAAPcv(train, folds, Om, gam, lams, q, PGsteps, PGtol, maxits, tol, feat, quiet);
T2 = toc (TAPG2);


% 
% T3 = 0;
% q = 1;
% PGsteps = 10;
% PGtol = 1e-5;
% maxits = 5;
% tol = 1e-3;
% feat = 0.15;
% 
% lams = sort(lams, 'descend');
% Xt = train.X(1:20, :);
% Yt = train.Y(1:20, :);
% [B3, ~] = SDAAP(Xt, Yt, Om, gam, lams(6), q, PGsteps, PGtol, maxits, tol);

% Test prediction.
if norm(B2) > 1e-12
    B2 = B2/norm(B2);
end

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



%% Call SDAD (ADMM).

% Set parameters.
%Om = eye(p) + gam*ones(p);
% Om = eye(p);
% gam = 0.001;
% lams = [0.001; 0.005; 0.015; 0.05; 0.15; 0.5; 5];
mu = 1;
q = 1;
%quiet = 0;
PGsteps = 1000;
Pt.abs = 1e-5;
Pt.rel = 1e-5;
maxits = 250;
tol = 1e-3;
%t0 = 10*randn(K, 1); t0 =t0/norm(t0);

folds = 5;

% Call solver
display('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
display('SDAD = SDA + ADMM')
display('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
TSMW = tic;
[B, ~, ~] = SDADcv(train, folds, Om, gam, lams, mu, q, PGsteps, Pt, maxits, tol, feat, quiet);
T3 =toc(TSMW);

% Extract solution.
B3 = B(:,:);
B3 = B3/norm(B3);

% plot(B); hold('on')
% plot(B2, 'r');
% hold('off')
% fprintf('norm(B) %3.3e\nnnz(B) %g\nTime %g\n', norm(B2), nnz(B2), T2)


% Test prediction.

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


%% Call SZVD.

%+++++++++++++++++++++++++++++++++++++++
% Initialize input.
%+++++++++++++++++++++++++++++++++++++++
% Make Atrain.
Atrain = zeros(n, p+1);
for i = 1:n
    for j = 1:K
        if (Y(i,j) == 1)
            Atrain(i,1) = j;
        end
    end
end
Atrain(:, 2:(p+1)) = X;

%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Compute penalty term for estimating range of regularization
% parameter values.
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

% Call ZVD function to solve the unpenalized problem.
get_DVs = 1;
w0 = ZVD(Atrain, 1, get_DVs);
%

% Set dictionary matrix.
D = eye(p);


% Normalize B (divide by the spectral norm)
if (size(w0.B, 2)==1) % B stored as vector.
    w0.B = w0.B/norm(w0.B);
else % B stored as p by p symmetric matrix.
    w0.B = w0.B + w0.B'; % symmetrize.
    w0.B =w0.B/norm(w0.B); %normalize.
    
end

% Compute ratio of max gen eigenvalue and l1 norm of the first ZVD to get "bound" on gamma.
if (size(w0.B, 2)==1)% B stored as vector.
    max_gamma =  ((w0.dvs)'*w0.B)^2/sum(abs(D * w0.dvs));
    
else % B stored as matrix.
    
    % Allocate gammas.
    max_gamma = zeros(q, 1);
    
    % Define max gamma to be value forcing the objective equal to zero
    % for each zero-variance DV.
    for i = 1:q
        max_gamma(i) = w0.dvs(:,i)'*w0.B*w0.dvs(:,i)/norm(D*w0.dvs(:,i), 1);
    end
    
end

%gams = 2.^(-10:1:2)*max_gamma;
gams = linspace(0,1, 15)*max_gamma;
gams = gams';
gams = sort(gams, 'descend');


%

% Set parameters.
%Om = eye(p) + gam*ones(p);
% Om = eye(p);
% gam = 0.001;
% lams = [0.001; 0.005; 0.015; 0.05; 0.15; 0.5; 5];
beta = 1.25;
q = 1;
feat = 0.35;
quiet = 1;
maxits = 1000;
Pt.abs = 1e-8;
Pt.rel = 1e-8;
ztol = 1e-5;

folds = 5;
%quiet =0;

% Call solver
display('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
display('SZVD')
display('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
TSZVD = tic;
[B4, best_ind] = SZVDcv(train, folds, gams(:), beta, D, q, maxits, Pt, ztol, feat, quiet);
T4 =toc(TSZVD);

% Round small entries to zero.
B5 = B4.*(ceil(abs(B4) - ztol));
% for j = 1:q
%     for k = 1:p
%         if abs(B4(k, j)) < 1e-7
%             B5(k,j) = 0;
%         end
%     end
% end
B5 = B5/norm(B5);            

%
% plot(B); hold('on')
% plot(B2, 'r');
% hold('off')
% fprintf('norm(B) %3.3e\nnnz(B) %g\nTime %g\n', norm(B2), nnz(B2), T2)
%[B4, B5];
% Test prediction.

B4 = B5;
B4 = B4.*(ceil(abs(B4) - 1e-5));

% Project test data.
PXtest = Xtest*B4;
% Project centroids.
PC = C*B4;

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
fprintf('norm(B) %3.3e\nnnz(B) %g\nTime %g \n', norm(B4), nnz(B4)/p/q, T4)
fprintf('# MC = %g\n', 1/2*norm(Ytest - Ypred, 'fro')^2)
fprintf('percent MC = %g\n\n', 1/2*norm(Ytest - Ypred, 'fro')^2/nt)


%% Call SDA.

% Set parameters.
%Om = eye(p) + gam*ones(p);
% Om = eye(p);
% gam = 0.001;
% lams = [0.001; 0.005; 0.015; 0.05; 0.15; 0.5; 5];
mu = 1;
q = 1;
%quiet = 0;
PGsteps = 1000;

feat = 0.15;
feats = linspace(0,1, 15)*q*p*feat;

maxits = 250;
tol = 1e-3;
%t0 = 10*randn(K, 1); t0 =t0/norm(t0);

folds = 5;

% Call solver
display('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
display('SDA')
display('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
TSDA = tic;
[B, Q, lbest] = SDAcv(train, folds, gam, feats, q, maxits, tol, quiet);
T5 =toc(TSDA);

% Extract solution.
B5 = B(:,:);
B5 = B5/norm(B5);

% plot(B); hold('on')
% plot(B2, 'r');
% hold('off')
% fprintf('norm(B) %3.3e\nnnz(B) %g\nTime %g\n', norm(B2), nnz(B2), T2)


% Test prediction.

% Project test data.
PXtest = Xtest*B5;
% Project centroids.
PC = C*B5;

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
fprintf('norm(B) %3.3e\nnnz(B) %g\nTime %g \n', norm(B5), nnz(B5)/p/q, T5)
fprintf('# MC = %g\n', 1/2*norm(Ytest - Ypred, 'fro')^2)
fprintf('percent MC = %g\n\n', 1/2*norm(Ytest - Ypred, 'fro')^2/nt)



%% Plots.

plot(B1); hold('on')
plot(B2, 'r');
plot(B3, 'g');
plot(B4, 'k');
plot(B5, 'p');
legend('PG', 'APG', 'ADMM', 'SZVD', 'SDA')
hold('off')


