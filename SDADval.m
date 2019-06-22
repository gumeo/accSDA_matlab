function [B, Q, best_ind, scores] = SDADval(train, val, Om, gam, lams, mu, q, PGsteps, PGtol, maxits, tol, feat, quiet)
% Applies alternating direction method of multipliers with validation
% to the optimal scoring formulation of
% sparse discriminant analysis proposed by Clemmensen et al. 2011.
%
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Input
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% train,val.Y: n by K matrix of indicator variables (Yij = 1 if i in classs j)
% train,val.X: n by p data matrix.
% Om: p by p parameter matrix Omega in generalized elastic net penalty.
% gam > 0: regularization parameter for elastic net penalty.
% lams > 0: vector of regularization parameters for l1 penalty.
% mu > 0: augmented Lagrangian penalty parameter used in ADMM step.
% q: desired number of discriminant vectors.
% PGsteps: max its of inner prox-grad algorithm to update beta.
% maxits: number of iterations to run alternating direction alg.
% tol: stopping tolerance for alternating direction algorithm.
%
% feat: maximum fraction of nonzero features desired in validation scheme.
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Output
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% B: p by q by nlam matrix of discriminant vectors.
% Q: K by q by nlam matrix of scoring vectors.
% best_ind: index of best solution in [B,Q].

%% Initialization.

% Sort lambdas in ascending order (break ties by using largest lambda =
% sparsest vector).
lams = sort(lams, 'ascend');

% Extract X and Y from train.
X = train.X;
[X, mut, sigt] = normalize(X);
Y = train.Y;
% [~, labs] = max(Y, [],2 );

% Get dimensions of input matrices.
[n, p] = size(X);
[~, K] = size(Y);

% Centroid matrix of training data.
C = diag(1./diag(Y'*Y))*Y'*X;

% Number of validation observations.
Xv = val.X;
Xv = normalize_test(Xv, mut, sigt);
[~, vlabs] = max(val.Y, [],2 );
% [nval,~] = size(val.X);

%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Precompute repeatedly used matrix products
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

%-------------------------------------------------------------------
% Matrices for ADMM step.
%-------------------------------------------------------------------

% Check if Om is diagonal. If so, use matrix inversion lemma in linear
% system solves.
if norm(diag(diag(Om)) - Om, 'fro') < 1e-15
    
    % Flag to use Sherman-Morrison-Woodbury to translate to
    % smaller dimensional linear system solves.    
    SMW = 1;
    
    % Easy to invert diagonal part of Elastic net coefficient matrix.
    M = mu + 2*gam*diag(Om); 
    Minv = 1./M;
    
    
    % Cholesky factorization for smaller linear system.
    %min(diag(M))
    RS = chol(eye(n) + 2*X*((Minv/n).*X') );
    
else % Use Cholesky for solving linear systems in ADMM step.
    
    % Flag to not use SMW.
    SMW = 0;
    A = mu*eye(p) + 2*(X'*X/n + gam*Om); % Elastic net coefficient matrix.
    R2 = chol(A); % Cholesky factorization of mu*I + A.
end

%-------------------------------------------------------------------
% Matrices for theta update.
%-------------------------------------------------------------------
D = 1/n*(Y'*Y); %D 
%M = X'*Y; % X'Y.
R = chol(D); % Cholesky factorization of D.


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
        
        % Initialize coefficient vector for elastic net step.
        d = 2*X'*(Y*(theta/n));
        
        % Initialize beta.
        if SMW == 1
            btmp = X*(Minv.*d)/n;
            beta = (Minv.*d) - 2*Minv.*(X'*(RS\(RS'\btmp)));
        else
            beta = R2\(R2'\d);
        end
        
        %+++++++++++++++++++++++++++++++++++++++++++++++++++++
        % Alternating direction method to update (theta, beta)
        %+++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        for its = 1:maxits
            
            % Update beta using alternating direction method of multipliers. 
            b_old = beta;
            
            if SMW == 1
                [~,beta,~,~] = ADMM_EN_SMW(Minv, X, RS, d, beta, lams(ll), mu, PGsteps, PGtol, 1);
            else
                [~, beta,~, ~] = ADMM_EN2(R2, d, beta, lams(ll), mu, PGsteps, PGtol, 1);
            end
       
            
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
    %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    % Get classification statistics for (Q,B).
    %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    
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

