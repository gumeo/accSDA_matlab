function [B, Q, subits, totalits] = SDAD(Xt, Yt, Om, gam, lam, mu, q, PGsteps, PGtol, maxits, tol, quiet)
% SDAD admm for the SOS problem.
% Applies alternating direction method of multipliers
% to the optimal scoring formulation of
% sparse discriminant analysis proposed by Clemmensen et al. 2011.
%
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% INPUT.
% Xt: n by p training data matrix.
% Yt: n by K matrix of indicator variables (Yij = 1 if i in classs j)
% folds: number of folds to use in K-fold cross-validation.
% Om: p by p parameter matrix Omega in generalized elastic net penalty.
% gam > 0: regularization parameter for elastic net penalty.
% lams > 0: regularization parameter for l1 penalty.
% mu > 0: penalty parameter for augmented Lagrangian term.
% q: desired number of discriminant vectors.
% PGsteps: max its of inner prox-grad algorithm to update beta.
% maxits: number of iterations to run alternating direction alg.
% PGtol: stopping tolerance for ADMM algorithm.
% tol: stopping tolerance for outer loop.
% quiet: toggle display of intermediate output.
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% OUTPUT.
% B: p by q by nlam matrix of discriminant vectors.
% Q: K by q by nlam matrix of scoring vectors.
% subits, totalits: number of inner and outer loop iterations.

% Record number of subproblem iterations.
subits = 0;
totalits = maxits*ones(q,1);

%% Initialize training sets, etc.

% Get dimensions of training matrices.
[nt, p] = size(Xt);
[~, K] = size(Yt);


% Check if Om is diagonal. If so, use matrix inversion lemma in linear
% system solves.
if norm(diag(diag(Om)) - Om, 'fro') < 1e-15
    
    % Flag to use Sherman-Morrison-Woodbury to translate to
    % smaller dimensional linear system solves.
    %display('Using SMW')
    SMW = 1;
    if quiet== false
        fprintf('Using SMW for x-update.\n')
    end
    
    % Easy to invert diagonal part of Elastic net coefficient matrix.
    M = mu + 2*gam*diag(Om); 
    Minv = 1./M;
    %fprintf('Minv err: %g\n', norm(diag(Minv) - inv(M)))
    %fprintf('max Minv: %g\n', max(Minv))
    
    % Cholesky factorization for smaller linear system.
    %min(diag(M))
    RS = chol(eye(nt) + 2*Xt*((Minv/nt).*Xt') );
    %fprintf('Chol norm: %g\n', norm(RS, 'fro'))
    
    % Coefficient matrix (Minv*X) = V*A^{-1} = (A^{-1}U)' in SMW.
    %XM = X*Minv;
 
    
else % Use Cholesky for solving linear systems in ADMM step.
    
    % Flag to not use SMW.
    SMW = 0;
    if quiet== false
        fprintf('Not using SMW for x-update.\n')
    end
    A = mu*eye(p) + 2*(Xt'*Xt/nt + gam*Om); % Elastic net coefficient matrix.
    R2 = chol(A); % Cholesky factorization of mu*I + A.
end

%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Matrices for theta update.
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
D = 1/nt*(Yt'*Yt); %D
%M = X'*Y; % X'Y.
R = chol(D); % Cholesky factorization of D.

% Initialize B and Q.
Q = ones(K,q);
B = zeros(p, q);

%%
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Call Alternating Direction Method to solve SDA.
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% For j=1,2,..., q compute the SDA pair (theta_j, beta_j).
%[f, lams(ll)]
for j = 1:q
    
    %+++++++++++++++++++++++++++++++++++++++++++++++++++++
    % Initialization.
    %+++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    % Compute Qj (K by j, first j-1 scoring vectors, all-ones last col).
    Qj = Q(:, 1:j);
    
    % Precompute Mj = I - Qj*Qj'*D.
    Mj = @(u) u - Qj*(Qj'*(D*u));
    
    % Initialize theta.
    theta = Mj(rand(K,1));
    %theta = Mj(theta0);
    theta = theta/sqrt(theta'*D*theta);
    
    % Initialize coefficient vector for elastic net step.
    d = 2*Xt'*(Yt*theta/nt);
    
    % Initialize beta.
    if SMW == 1
        btmp = Xt*(Minv.*d)/nt;
        beta = (Minv.*d) - 2*Minv.*(Xt'*(RS\(RS'\btmp)));
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
            % Use SMW-based ADMM.
            
            [~,beta,~, steps] = ADMM_EN_SMW(Minv, Xt, RS, d, beta, lam, mu, PGsteps, PGtol, quiet);
        else
            % Use vanilla ADMM.
            [~, beta,~, steps] = ADMM_EN2(R2, d, beta, lam, mu, PGsteps, PGtol, quiet);
        end
        subits = subits + steps;
                
        
        % Update theta using the projected solution.
        % theta = Mj*D^{-1}*Y'*X*beta.
        if norm(beta) > 1e-15
            % update theta.
            b = Yt'*(Xt*beta);
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
        end
        
        % Print progress.
        if quiet == false
            fprintf('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
            fprintf('OutIt: %1.2d \t + db %1.2e \t + dt %5.2e \n', its, db, dt)            
        end
        
        
         % Check convergence.        
        if max(db, dt) <= tol             % Converged.
            totalits(j) = its;
            if quiet == false
                fprintf('Found discriminant vector %d after %d iterations.\n', j, its)
            end
            break
        end
    end %its
    
    % Update Q and B.
    Q(:,j) = theta;
    B(:,j) = beta;
end %j.

% Sum iterations.
totalits = sum(totalits);

