% These are tests to compare to results from R

%% Test prox_EN algorithm 

% Generate some input
a = [1,2,3;2,3,1;5,1,2];
A = a'*a;
d = [1;2;3];
x0 = [1.5;3;2.2];
lam = 1;
alpha = 0.001;
maxits = 50;
tol = 0.0000001;

% Run the method
prox_EN(A, d, x0, lam, alpha, maxits, tol)

% R generates same results

%% Test SDAP algorithm
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
lam = 0.1;
q = 2;
PGsteps = 1000;
PGtol = 1e-5;
maxits = 10; % Just a low number to test
tol = 1e-3;

[B,Q] = SDAP(Xt, Yt, Om, gam, lam, q, PGsteps, PGtol, maxits, tol)

% R generates same results

%% Test SDAPcv algorithm
clear all;clc;

% Test on iris data set
load fisheriris;
Xt = meas;

% generate Yt
Yt = zeros(150,3);
Yt(1:50,1) = ones(50,1);
Yt(51:100,2) = ones(50,1);
Yt(101:150,3) = ones(50,1);
n = 150;
Om = eye(4);
gam = 0.001;
K = 3;
% Form A.
A = 2*(Xt'*Xt/n + gam*Om);

Qj = ones(K, 1);
D = 1/n*(Yt'*Yt);
Mj = @(u) u - Qj*(Qj'*(D*u));

% Initialize theta.
theta = Mj(rand(K,1));
theta = theta/sqrt(theta'*D*theta);

% Form d.
d = 2*Xt'*Yt*theta/n;

% Initialize beta.
beta = A\d; % 1st unpenalized solution.

% Choose lambda so that unpenalized solution always has negative value.
lmax = (beta'*d - 0.5*beta'*A*beta)/norm(beta, 1);

lams = 2.^(-9:1:-2)*lmax;

q = 2;
PGsteps = 1000;
PGtol = 1e-5;
maxits = 10; % Just a low number to test
tol = 1e-3;
folds = 5;
train.X = Xt;
train.Y = Yt;
feat = 0.15;
quiet = 0;
[B, Q, lbest, lambest]=SDAPcv(train, folds, Om, gam, lams, q, PGsteps, PGtol, maxits, tol, feat, quiet);

% Matlab and R are outputting the same solution

%% Test APG_EN2
clc, clear all, close all;

% Test on iris data set
load fisheriris;
Xt = meas(:,1:3);
[nt,~] = size(Xt);
gam = 0.1;
Om = diag(1:1:3);

if norm(diag(diag(Om)) - Om, 'fro') < 1e-15 % Omega is diagonal.
    A.flag = 1;
    % Store components of A.
    A.gom = gam*diag(Om);
    A.X = Xt;
    A.n = nt;
    
    alpha = 1/( 2*(norm(Xt,1)*norm(Xt,'inf')/nt + norm(A.gom, 'inf') ));
else
    A.flag = 0;
    A.A = 2*(Xt'*Xt/nt + gam*Om); % Elastic net coefficient matrix.
    alpha = 1/norm(A.A, 'fro');
end

d = [1;2;3];
x0 = [1.5;3;2.2];
lam = 1;
alpha = 0.001;
maxits = 50;
tol = 0.0000001;

APG_EN2(A, d, x0, lam, alpha, maxits, tol)

% Matlab and R produce same results

%% Test SDAAP
clear all;clc;

% Test on iris data set
load fisheriris;
Xt = meas;

% normalize
n = 150;
mu = mean(Xt)';
%muz = size(mu);
Xt = Xt - ones(n,1)* mu';

% Scale the data so each feature has variance equal to 1.

% Compute standard deviation of each feature.
sig = std(Xt);
In = find(sig~=0);

% Divide each feature by its standard deviation to ensure each feature has variance equal to one.
Xt = Xt(:,In) *diag(1./sig(In));
%done normalizing

% generate Yt
Yt = zeros(150,3);
Yt(1:50,1) = ones(50,1);
Yt(51:100,2) = ones(50,1);
Yt(101:150,3) = ones(50,1);

Om = eye(4);
gam = 0.001;

% Some value, maybe change it
lam = 0.1;
q = 2;
PGsteps = 1000;
PGtol = 1e-5;
maxits = 10; % Just a low number to test
tol = 1e-3;

[B,Q] = SDAAP(Xt, Yt, Om, gam, lam, q, PGsteps, PGtol, maxits, tol)

% R generates same results, up to a scaling factor of -1, which
% is due to random initialization in the function

%% Test SDAAPcv algorithm
clear all;clc;

% Test on iris data set
load fisheriris;
Xt = meas;

% generate Yt
Yt = zeros(150,3);
Yt(1:50,1) = ones(50,1);
Yt(51:100,2) = ones(50,1);
Yt(101:150,3) = ones(50,1);
n = 150;
Om = eye(4);
gam = 0.001;
K = 3;
% Form A.
A = 2*(Xt'*Xt/n + gam*Om);

Qj = ones(K, 1);
D = 1/n*(Yt'*Yt);
Mj = @(u) u - Qj*(Qj'*(D*u));

vec = [0.5;0.55;0.45];

% Initialize theta.
theta = Mj(vec);
theta = theta/sqrt(theta'*D*theta);

% Form d.
d = 2*Xt'*Yt*theta/n;

% Initialize beta.
beta = A\d; % 1st unpenalized solution.

% Choose lambda so that unpenalized solution always has negative value.
lmax = (beta'*d - 0.5*beta'*A*beta)/norm(beta, 1);

lams = 2.^(-15:1:3)*lmax;

q = 2;
PGsteps = 1000;
PGtol = 1e-5;
maxits = 50; % Just a low number to test
tol = 1e-3;
folds = 15;
train.X = Xt;
train.Y = Yt;
feat = 0.15;
quiet = 0;
[B, Q, lbest, lambest]=SDAAPcv(train, folds, Om, gam, lams, q, PGsteps, PGtol, maxits, tol, feat, quiet)

% Matlab and R are outputting the same solution

%% Test ADMM_EN_SMW algorithm
clear all;clc;

% Test on iris data set
load fisheriris;
Xt = meas;

% generate Yt
Yt = zeros(150,3);
Yt(1:50,1) = ones(50,1);
Yt(51:100,2) = ones(50,1);
Yt(101:150,3) = ones(50,1);
[n, p] = size(Xt);
nt = n;
[~, K] = size(Yt);
Om = eye(4);
mu = 1.1;
gam = 0.001;


if norm(diag(diag(Om)) - Om, 'fro') < 1e-15
    % Flag to use Sherman-Morrison-Woodbury to translate to
    % smaller dimensional linear system solves.
    SMW = 1;
    M = mu*eye(p) + 2*gam*Om;
    Minv = 1./diag(M);
    
    % Cholesky factorization for smaller linear system.
    RS = chol(eye(nt) + 2*Xt*diag(Minv)*Xt'/nt);
else % Use Cholesky for solving linear systems in ADMM step.
    % Flag to not use SMW.
    SMW = 0;
    A = mu*eye(p) + 2*(Xt'*Xt + gam*Om); % Elastic net coefficient matrix.
    R2 = chol(A); % Cholesky factorization of mu*I + A.
end

D = 1/nt*(Yt'*Yt); %D
R = chol(D); % Cholesky factorization of D.

q = 2;
j = 1;

% Initialize B and Q.
Q = ones(K,q);
B = zeros(p, q);


Qj = Q(:, 1:j);
    
% Precompute Mj = I - Qj*Qj'*D.
Mj = @(u) u - Qj*(Qj'*(D*u));

% Initialize theta.
theta = Mj(rand(K,1));
%theta = Mj(theta0);
theta = theta/sqrt(theta'*D*theta);

% Initialize coefficient vector for elastic net step.
d = 2*Xt'*(Yt*theta);

% Initialize beta.
if SMW == 1
    btmp = Xt*(Minv.*d)/nt;
    beta = (Minv.*d) - 2*Minv.*(Xt'*(RS\(RS'\btmp)));
else
    beta = R2\(R2'\d);
end

PGsteps = 1000;
PGtol.abs = 1e-5;
PGtol.rel = 1e-5;
lam = 0.1;

[x,y,z, k] = ADMM_EN_SMW(Minv, Xt,RS, d, beta, lam, mu, PGsteps, PGtol, 1)

% Matlab and R are outputting the same solution up to initial randomization

%% ADMM_EN2
clear all;clc;

% Test on iris data set
load fisheriris;
Xt = meas;

% generate Yt
Yt = zeros(150,3);
Yt(1:50,1) = ones(50,1);
Yt(51:100,2) = ones(50,1);
Yt(101:150,3) = ones(50,1);
[n, p] = size(Xt);
nt = n;
[~, K] = size(Yt);
Om = eye(4) + 0.1*ones(4,4);
mu = 1.1;
gam = 0.001;


if norm(diag(diag(Om)) - Om, 'fro') < 1e-15
    % Flag to use Sherman-Morrison-Woodbury to translate to
    % smaller dimensional linear system solves.
    SMW = 1;
    M = mu*eye(p) + 2*gam*Om;
    Minv = 1./diag(M);
    
    % Cholesky factorization for smaller linear system.
    RS = chol(eye(nt) + 2*Xt*diag(Minv)*Xt'/nt);
else % Use Cholesky for solving linear systems in ADMM step.
    % Flag to not use SMW.
    SMW = 0;
    A = mu*eye(p) + 2*(Xt'*Xt + gam*Om); % Elastic net coefficient matrix.
    R2 = chol(A); % Cholesky factorization of mu*I + A.
end

D = 1/nt*(Yt'*Yt); %D
R = chol(D); % Cholesky factorization of D.

q = 2;
j = 1;

% Initialize B and Q.
Q = ones(K,q);
B = zeros(p, q);


Qj = Q(:, 1:j);
    
% Precompute Mj = I - Qj*Qj'*D.
Mj = @(u) u - Qj*(Qj'*(D*u));

% Initialize theta.
theta = Mj(rand(K,1));
%theta = Mj(theta0);
theta = theta/sqrt(theta'*D*theta);

% Initialize coefficient vector for elastic net step.
d = 2*Xt'*(Yt*theta);

% Initialize beta.
if SMW == 1
    btmp = Xt*(Minv.*d)/nt;
    beta = (Minv.*d) - 2*Minv.*(Xt'*(RS\(RS'\btmp)));
else
    beta = R2\(R2'\d);
end

PGsteps = 1000;
PGtol.abs = 1e-15;
PGtol.rel = 1e-15;
lam = 0.1;

[x,y,z, k] = ADMM_EN2(R2, d, beta, lam, mu, PGsteps, PGtol, 1)

% Outputs are the same as in R

%% SDAD
clear all;clc;

% Test on iris data set
load fisheriris;
Xt = meas;

%normalize
n = 150;
mu = mean(Xt)';
%muz = size(mu);
Xt = Xt - ones(n,1)* mu';

% Scale the data so each feature has variance equal to 1.

% Compute standard deviation of each feature.
sig = std(Xt);
In = find(sig~=0);

% Divide each feature by its standard deviation to ensure each feature has variance equal to one.
Xt = Xt(:,In) *diag(1./sig(In));
% done normalizing

% generate Yt
Yt = zeros(150,3);
Yt(1:50,1) = ones(50,1);
Yt(51:100,2) = ones(50,1);
Yt(101:150,3) = ones(50,1);

Om = eye(4);
gam = 0.001;

% Some value, maybe change it
lam = 0.1;
q = 2;
PGsteps = 100;
PGtol.abs = 1e-17;
PGtol.rel = 1e-17;
maxits = 100; % Just a low number to test
tol = 1e-15;
mu = 0.1;

[B,Q] = SDAD(Xt, Yt, Om, gam, lam, mu, q, PGsteps, PGtol, maxits, tol)

% R generates same results, up to a scaling factor of -1, which
% is due to random initialization in the function

%% SDADcv
clear all;clc;

% Test on iris data set
load fisheriris;
Xt = meas;

% generate Yt
Yt = zeros(150,3);
Yt(1:50,1) = ones(50,1);
Yt(51:100,2) = ones(50,1);
Yt(101:150,3) = ones(50,1);
n = 150;
Om = eye(4);
gam = 0.001;
K = 3;
% Form A.
A = 2*(Xt'*Xt/n + gam*Om);

Qj = ones(K, 1);
D = 1/n*(Yt'*Yt);
Mj = @(u) u - Qj*(Qj'*(D*u));

vec = [0.5;0.55;0.45];

% Initialize theta.
theta = Mj(vec);
theta = theta/sqrt(theta'*D*theta);

% Form d.
d = 2*Xt'*Yt*theta/n;

% Initialize beta.
beta = A\d; % 1st unpenalized solution.

% Choose lambda so that unpenalized solution always has negative value.
lmax = (beta'*d - 0.5*beta'*A*beta)/norm(beta, 1);

lams = 2.^(-15:1:7)*lmax;

q = 2;
PGsteps = 1000;
PGtol.abs = 1e-5;
PGtol.rel = 1e-5;
maxits = 50; % Just a low number to test
tol = 1e-3;
folds = 15;
train.X = Xt;
train.Y = Yt;
feat = 0.15;
quiet = 0;
mu = 0.1;

[B, Q, lbest, lambest]=SDADcv(train, folds, Om, gam, lams, mu, q, PGsteps, PGtol, maxits, tol, feat, quiet)

% This function seems to be giving the same results as R. It is not
% consistent in the columns of B, they seem to span the same subspace, but
% it is not consistent which ones come out.

%% Test to see how SDAP handles theta as 0

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
lam = 5;
q = 2;
PGsteps = 1000;
PGtol = 1e-5;
maxits = 10; % Just a low number to test
tol = 1e-3;

[B,Q] = SDAP(Xt, Yt, Om, gam, lam, q, PGsteps, PGtol, maxits, tol)

%% Test slda to understand orth_theta function

clear all;clc;

% Test on iris data set
load fisheriris;
Xt = meas;

% generate Yt
Yt = zeros(150,3);
Yt(1:50,1) = ones(50,1);
Yt(51:100,2) = ones(50,1);
Yt(101:150,3) = ones(50,1);

Y = Yt;