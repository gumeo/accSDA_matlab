function [B, Q] = ADMM_SDA2(t0, Y, X, Om, gam, lam, mu, q, PGsteps, PGtol, maxits, tol)

% Applies proximal gradient algorithm to the optimal scoring formulation of
% sparse discriminant analysis proposed by Clemmensen et al. 2011.
%
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Input
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Y: n by K matrix of indicator variables (Yij = 1 if i in classs j)
% X: n by p data matrix.
% Om: p by p parameter matrix Omega in generalized elastic net penalty.
% gam > 0: regularization parameter for elastic net penalty.
% lam > 0: regularization parameter for l1 penalty.
% mu > 0: augmented Lagrangian penalty parameter used in ADMM step.
% q: desired number of discriminant vectors.
% PGsteps: max its of inner ADMM algorithm to update beta.
% maxits: number of iterations to run alternating direction alg.
% tol: stopping tolerance for alternating direction algorithm.
%
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Output
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% B: p by q matrix of discriminant vectors.
% Q: K by q matrix of scoring vectors.



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Get dimensions of input matrices.
[n, p] = size(X);
[~, K] = size(Y);

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
    display('Using SMW')
    SMW = 1;
    
    % Easy to invert diagonal part of Elastic net coefficient matrix.
    M = mu*eye(p) + 2*gam*Om;
    fprintf('min M: %g\n', min(diag(M)))
    Minv = 1./diag(M);
    %fprintf('Minv err: %g\n', norm(diag(Minv) - inv(M)))
    fprintf('max Minv: %g\n', max(Minv))
    
    % Cholesky factorization for smaller linear system.
    %min(diag(M))
    RS = chol(eye(n) + 2*X*diag(Minv)*X'/n);
    fprintf('Chol norm: %g\n', norm(RS, 'fro'))
    
    % Coefficient matrix (Minv*X) = V*A^{-1} = (A^{-1}U)' in SMW.
    %XM = X*Minv;
    
else % Use Cholesky for solving linear systems in ADMM step.
    
    % Flag to not use SMW.
    SMW = 0;
    A = mu*eye(p) + 2*(X'*X + gam*Om); % Elastic net coefficient matrix.
    R2 = chol(A); % Cholesky factorization of mu*I + A.
end

%-------------------------------------------------------------------
% Matrices for theta update.
%-------------------------------------------------------------------
D = 1/n*(Y'*Y); %D 
%M = X'*Y; % X'Y.
R = chol(D); % Cholesky factorization of D.

%-------------------------------------------------------------------
% Initialize B and Q.
%-------------------------------------------------------------------
Q = ones(K,q);
B = zeros(p, q);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Outer loop.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% For j=1,2,..., q compute the SDA pair (theta_j, beta_j).

for j = 1:q
    
    %+++++++++++++++++++++++++++++++++++++++++++++++++++++
    % Initialization.
    %+++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    % Compute Qj (K by j, first j-1 scoring vectors, all-ones last col).
    Qj = Q(:, 1:j);
    
    % Precompute Mj = I - Qj*Qj'*D.
    Mj = @(u) u - Qj*(Qj'*(D*u));
    
    % Initialize theta.
    theta = Mj(t0);
    %Mjt = theta;
    theta = theta/sqrt(theta'*D*theta);
    
    % Initialize coefficient vector for elastic net step.
    d = 2*X'*(Y*theta);
    
    % Initialize beta.
    if SMW == 1
        btmp = X*(Minv.*d)/n;
        beta = (Minv.*d) - 2*Minv.*(X'*(RS\(RS'\btmp)));
    else
        beta = R2\(R2'\d);        
    end
    
    %fprintf('max lambda %g\n', norm(beta, 'inf'));
    
    %+++++++++++++++++++++++++++++++++++++++++++++++++++++
    % Alternating direction method to updata (theta, beta)
    %+++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    for its = 1:maxits
        
        % Update beta using alternating direction method of multipliers. 
        b_old = beta;
        
        if SMW == 1
            [~,beta,~,subprob_its] = ADMM_EN_SMW(Minv, X, RS, d, beta, lam, mu, PGsteps, PGtol, 1);
        else
            [~, beta,~, subprob_its] = ADMM_EN2(R2, d, beta, lam, mu, PGsteps, PGtol, 1);
        end
       
        
        % Update theta using the projected solution.
        % theta = Mj*D^{-1}*Y'*X*beta.
       if norm(beta) > 1e-15
            % update theta.
            b = Y'*(X*beta);
            y = R'\b;
            z = R\y;
            tt = Mj(z);
            t_old = theta;
            theta = tt/sqrt(tt'*D*tt);
            
            % Update changes..
            db = norm(beta-b_old)/norm(beta);
            dt = norm(theta-t_old)/norm(theta);
            
        else
            % Update b and theta.
            beta = 0;
            theta = 0;
            % Update change.
            db = 0;
            dt = 0;
        end;        
        
        fprintf('It %5.0f   nb %5.2e   db %5.2e      dt %5.2e      Subprob its %5.0f\n', its, norm(beta), db, dt, subprob_its)
        
        % Check convergence.
        if max(db, dt) < tol
            % Converged.
            fprintf('Found %d-th dv after %g iterations\n\n', j, its)
            break
        end
    end
    
    % Update Q and B.
    Q(:,j) = theta;
    B(:,j) = beta;
end

