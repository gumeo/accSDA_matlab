function [B, Q, best_ind, scores] = SDAPbtval(train, val, Om, gam, lams, L, eta, q, PGsteps, PGtol, maxits, tol, feat, quiet)
% SDAPBTVAL PG with backtracking line search and cross validation for SOS.
% Applies accelerated proximal gradient algorithm with validation
% to the optimal scoring formulation of
% sparse discriminant analysis proposed by Clemmensen et al. 2011.
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% INPUT.
% train,val.Y: n by K matrix of indicator variables (Yij = 1 if i in classs j)
% train,val.X: n by p data matrix.
% Om: p by p parameter matrix Omega in generalized elastic net penalty.
% gam > 0: regularization parameter for elastic net penalty.
% lams > 0: vector of regularization parameters for l1 penalty.
% q: desired number of discriminant vectors.
% PGsteps: max its of inner prox-grad algorithm to update beta.
% maxits: number of iterations to run alternating direction alg.
% tol: stopping tolerance for alternating direction algorithm.%
% feat: maximum fraction of nonzero features desired in validation scheme.
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% OUTPUT.
% B: p by q by nlam matrix of discriminant vectors.
% Q: K by q by nlam matrix of scoring vectors.
% best_ind: index of best solution in [B,Q].
% scores: matrix of scores.

%% Initialization.

% Sort lambdas in ascending order (break ties by using largest lambda =
% sparsest vector).
lams = sort(lams, 'ascend');

% Extract X and Y from train.
X = train.X;
[X, mut, sigt] = normalize(X);
Y = train.Y;

% Get dimensions of input matrices.
[n, p] = size(X);
[~, K] = size(Y);

% Centroid matrix of training data.
C = diag(1./diag(Y'*Y))*Y'*X;

% Number of validation observations.
Xv = val.X;
Xv = normalize_test(Xv, mut, sigt);
[~, vlabs] = max(val.Y, [],2 );

% Precompute repeatedly used matrix products
A = 2*(X'*X/n + gam*Om); % Elastic net coefficient matrix.
% alpha = 1/norm(A, 'fro'); % Step length in PGA.
D = 1/n*(Y'*Y); %D 
%XY = X'*Y; % X'Y.
R = chol(D);


%% Validation Loop.

% Get number of parameters to test.
nlam = length(lams);

% Initialize validation scores.
scores = zeros(nlam, 1);

% Position of best solution.
best_ind = 1;

% Misclassification rate for each classifier.
mc = zeros(nlam, 1);

% Initialize B and Q.
Q = ones(K,q, nlam);
B = zeros(p, q, nlam);



%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Loop through potential regularization parameters.
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
for ll = 1:nlam
    
    %%
    %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    % Call Alternating Direction Method to solve SDA.
    %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    
    % For j=1,2,..., q compute the SDA pair (theta_j, beta_j).
    for j = 1:q
        
        %+++++++++++++++++++++++++++++++++++++++++++++++++++++
        % Initialization.
        %+++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        % Compute Qj (K by j, first j-1 scoring vectors, all-ones last col).
        Qj = Q(:, 1:j, ll);
        
        % Precompute Mj = I - Qj*Qj'*D.
        Mj = @(u) u - Qj*(Qj'*(D*u));
        
        % Initialize theta.
        theta = Mj(rand(K,1));
        theta = theta/sqrt(theta'*D*theta);
        
        % Initialize beta.
        beta = zeros(p,1);        
        
        %+++++++++++++++++++++++++++++++++++++++++++++++++++++
        % Alternating direction method to update (theta, beta)
        %+++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        for its = 1:maxits
            
            % Compute coefficient vector for elastic net step.
            d = 2*X'*(Y*(theta/n));
            
            % Update beta using proximal gradient step.
            b_old = beta;
            
            [beta, ~] = prox_ENbt(A, d, beta, lams(ll), L,eta, PGsteps, PGtol);
            
            
            % Update theta using the projected solution.
            % theta = Mj*D^{-1}*Y'*X*beta.
            b = Y'*(X*beta);
            y = R'\b;
            z = R\y;
            tt = Mj(z);
            t_old = theta;
            theta = tt/sqrt(tt'*D*tt);
            
            % Progress.
            db = norm(beta-b_old)/norm(beta);
            dt = norm(theta-t_old)/norm(theta);
            %fprintf('It %5.0f      db %5.2f      dt %5.2f      Subprob its %5.0f It time %5.2f\n', its, db, dt, subprob_its, update_time)
            
            % Check convergence.
             if max(db, dt) < tol
                 % Converged.
                 %fprintf('Algorithm converged after %g iterations\n\n', its)
                 break
             end
        end
        
        % Update Q and B.
        Q(:,j, ll) = theta;
        B(:,j, ll) = beta;
    end
    
    %%
    % Get prediction/scores.
    stats = predict(B(:,:,ll), [vlabs, Xv], C');
    mc(ll) = stats.mc;    
     
    if (1<= stats.l0) && (stats.l0 <= q*p*feat)        
        
        fprintf('Sparse enough. Use MC as score. \n')
        % Use misclassification rate as validation score.
        scores(ll) = mc(ll);

    elseif (stats.l0 > q*p*feat) % Solution is not sparse enough, use most sparse as measure of quality instead.
        fprintf('Not sparse enough. Use cardinality as score. \n')
        
        scores(ll) = stats.l0;
    end    
    
        
    % Update best so far.
    if (scores(ll) <= scores(best_ind))
        best_ind = ll;
    end
    
    % Display iteration stats.
    if (quiet ==0)
        fprintf('ll: %d | lam: %1.2e| feat: %d | mc: %1.2e | score: %1.2e | best: %d\n', ll, lams(ll), nnz(B(:,:,ll)), mc(ll),scores(ll), best_ind)
    end    
    
    % Update best so far.
    if (scores(ll) <= scores(best_ind))
        best_ind = ll;
    end
    
     
    
    
    
    
    
    
end % For ll = 1:nlam.

% Output best solution when finished.
B = B(:,:, best_ind);
Q = Q(:,:, best_ind);

end % Function.

    
    
