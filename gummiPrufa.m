% Create training data for SZVD to verify whether we get something sparse
% here...
p = 300;
n = 50;
m1 = zeros(p,1);
m1(1) = 3;
m2 = zeros(p,1);
m2(2) = 3;
m3 = zeros(p,1);
m3(3) = 3;
train = [mvnrnd(m1,eye(p),50);...
         mvnrnd(m2,eye(p),50);...
         mvnrnd(m3,eye(p),50)];

train = [[repmat(1,n,1);repmat(2,n,1);repmat(3,n,1)] train];

tol.abs = 1e-4;
tol.rel=1e-4;
res = SZVD(train, [0.001 0.001], eye(p),0,1,tol,2000,2.5,0);
