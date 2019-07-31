function [B,Q, subits, totalits] = SDAAP(Xt, Yt, Om, gam, lam, q, PGsteps, PGtol, maxits, tol, quiet)
% Applies accelerated proximal gradient algorithm 
% to the optimal scoring formulation of
% sparse discriminant analysis proposed by Clemmensen et al. 2011.
%
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Input
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Xt: n by p data matrix.
% Yt: n by K matrix of indicator variables (Yij = 1 if i in classs j)
% Om: p by p parameter matrix Omega in generalized elastic net penalty.
% gam > 0: regularization parameter for elastic net penalty.
% lam > 0: regularization parameters for l1 penalty.
% q: desired number of discriminant vectors.
% PGsteps: max its of inner prox-grad algorithm to update beta.
% PGtol: stopping tolerance for inner APG method.
% maxits: number of iterations to run alternating direction alg.
% tol: stopping tolerance for alternating direction algorithm.
% quiet: toggle display of intermediate output.
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Output
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% B: p by q by nlam matrix of discriminant vectors.
% Q: K by q by nlam matrix of scoring vectors.


% Read training data size.
[nt, p] = size(Xt);
[~, K] = size(Yt);

% Record number of subproblem iterations.
subits = 0;
totalits = maxits*ones(q,1);


% Precompute repeatedly used matrix products
if norm(diag(diag(Om)) - Om, 'fro') < 1e-15 % Omega is diagonal.
    A.flag = 1;
    % Store components of A.
    A.gom = gam*diag(Om);
    A.X = Xt;
    A.n = nt;
    
    % Calculate step length as approximate Lipschitz constant.
    alpha = 1/( 2*(norm(Xt,1)*norm(Xt,'inf')/nt + norm(A.gom, 'inf') ));
else % Omegat is not diagonal. Explicitly calculate A. Use frobenius norm for step length.
    A.flag = 0;
    A.A = 2*(Xt'*Xt/nt + gam*Om); % Elastic net coefficient matrix.
    alpha = 1/norm(A.A, 'fro');
end

% Calculate D and Cholesky factor.
D = 1/nt*(Yt'*Yt); %D
R = chol(D);

% Initialize B and Q.
Q = ones(K,q);
B = zeros(p, q);

%+++++++++++++++++++++++++++++++++++++++++++++++++++++
% Alternating direction method to update (theta, beta)
%+++++++++++++++++++++++++++++++++++++++++++++++++++++
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
    theta = theta/sqrt(theta'*D*theta);
    
    % Initialize beta.
    if norm(diag(diag(Om)) - Om, 'fro') < 1e-15 % Use diagonal initializer.
        % Extract reciprocal of diagonal of Omega.
        ominv = 1./diag(Om);
        
        % Compute rhs of f minimizer system.
        rhs0 = Xt'*(Yt*(theta/nt));
        rhs = Xt*((ominv/nt).*rhs0);
        
        % Compute partial solution.
        tmp = (eye(nt) + Xt*((ominv/(gam*nt)).*Xt'))\rhs;
        
        % Finishing solving for beta using SMW.
        beta = (ominv/gam).*rhs0 - 1/gam^2*ominv.*(Xt'*tmp);      
        
    else
        % Initialize with all-zeros beta.
        beta = zeros(p,1);
    end
    
    
    
    for its = 1:maxits
        
        % Compute coefficient vector for elastic net step.
        d = 2*Xt'*(Yt*(theta/nt));
        
        % Update beta using proximal gradient step.
        b_old = beta;
        
        [beta, steps] = APG_EN2(A, d, beta, lam, alpha, PGsteps, PGtol, quiet);
        subits = subits + steps;
        
        % Update theta using the projected solution.
        % theta = Mj*D^{-1}*Y'*X*beta.        
        if norm(beta) > 1e-12
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
      
        
        % Progress.              
        if quiet == false
            fprintf('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
            fprintf('OutIt: %1.2d \t + db %1.2e \t\t + dt %5.2e \n', its, db, dt)            
        end
        
        % Check convergence.        
        if max(db, dt) <= tol             % Converged.
            totalits(j) = its;
            break
        end
    end
    
    % Update Q and B.
    Q(:,j) = theta;
    B(:,j) = beta;
end

% Sum iterations.
totalits = sum(totalits);

