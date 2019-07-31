function [B,Q, subits, totalits]  = SDAP(Xt, Yt, Om, gam, lam, q, PGsteps, PGtol, maxits, tol, quiet)
% Applies proximal gradien algorithm 
% to the optimal scoring formulation of
% sparse discriminan analysis proposed by Clemmensen et al. 2011.
%
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Input`
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Xt: n by p data matrix.
% Yt: n by K matrix of indicator variables (Yij = 1 if i in classs j)
% Om: p by p parameter matrix Omega in generalized elastic net penalty.
% gam > 0: regularization parameter for elastic net penalty.
% lam > 0: regularization parameters for l1 penalty.
% q: desired number of discriminan vectors.
% PGsteps: max its of inner prox-grad algorithm to update beta.
% PGtol: stopping tolerance for inner APG method.
% maxits: number of iterations to run alternating direction alg.
% tol: stopping tolerance for alternating direction algorithm.
% quiet: toggle display of intermediate output.
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Output
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% B: p by q by nlam matrix of discriminan vectors.
% Q: K by q by nlam matrix of scoring vectors.
% subits, totalits: number of inner and outer loop iterations.

% Record number of subproblem iterations.
subits = 0;
totalits = maxits*ones(q,1);


% Read training data size.
[n, p] = size(Xt);
[~, K] = size(Yt);

% Precompute repeatedly used matrix products

A = 2*(Xt'*Xt/n + gam*Om);
% fprintf('A norm: %d\n', norm(A))% Elastic net coefficien matrix.
alpha = 1/norm(A, 'fro'); % Step length in PGA.
D = 1/n*(Yt'*Yt); %D 
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
        rhs0 = Xt'*(Yt*(theta/n));
        rhs = Xt*((ominv/n).*rhs0);
        
        % Compute partial solution.
        tmp = (eye(n) + Xt*((ominv/(gam*n)).*Xt'))\rhs;
        
        % Finishing solving for beta using SMW.
        beta = (ominv/gam).*rhs0 - 1/gam^2*ominv.*(Xt'*tmp);      
        
    else
        % Initialize with all-zeros beta.
        beta = zeros(p,1);
    end
    
    
    
    %+++++++++++++++++++++++++++++++++++++++++++++++++++++
    % Alternating direction method to update (theta, beta)
    %+++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    for its = 1:maxits
        
        % Compute coefficien vector for elastic net step.
        d = 2*Xt'*(Yt*(theta/n));
        
        % Update beta using proximal gradien step.
        b_old = beta;

        [beta, steps] = prox_EN(A, d, beta, lam, alpha, PGsteps, PGtol, quiet);
        subits = subits + steps;
                
        % Update theta using the projected solution.
        % theta = Mj*D^{-1}*Y'*X*beta.
        b = Yt'*(Xt*beta);
        y = R'\b;
        z = R\y;
        tt = Mj(z);
        t_old = theta;
        theta = tt/sqrt(tt'*D*tt);
        
        % Progress.
        db = norm(beta-b_old)/norm(beta);
        dt = norm(theta-t_old)/norm(theta);
        if quiet == false
            fprintf('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
            fprintf('OutIt: %1.2d \t + db %1.2e \t\t + dt %5.2e \n', its, db, dt)            
        end
        
         % Check convergence.        
        if max(db, dt) <= tol             % Converged.
            totalits(j) = its;
            if quiet == false
                fprintf('Found discriminant vector %d after %d iterations.\n', j, its)
            end
            break
        end
    end
    
    % Update Q and B.
    Q(:,j) = theta;
    B(:,j) = beta;
end
    

% Sum iterations.
totalits = sum(totalits);