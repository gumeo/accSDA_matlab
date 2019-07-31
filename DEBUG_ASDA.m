X = [rand(4, 6); rand(4,6) + 2];
Y = [ones(4,1), zeros(4,1); zeros(4,1), ones(4,1)];


%%
Om = eye(6);
gam = 1e-3;
lam = 1e-2;

cv = false;

method = "APG";
opts.q = 1;
opts.insteps = 1500;
opts.outsteps = 10;
opts.intol = 1e-4;
opts.outtol = 1e-4;
opts.quiet = true;
opts.bt = true;
opts.L = 0.25;
opts.eta = 1.25;

%% 
[B,Q] = ASDA(X,Y, Om, gam, lam, cv, method, opts)
