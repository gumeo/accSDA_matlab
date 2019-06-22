function [x, k, L] = prox_ENbt(A, d, x0, lam,L, eta, maxits, tol)

% Applies proximal gradient algorithm to the l1-regularized quad
%   f(x) + g(x) = 0.5*x'*A*x - d'*x + lam*l1(x).
%
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Input
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% A: n by n positive definite coefficient matrix
% d: n dim coefficient vector.
% lam > 0: regularization parameter for l1 penalty.
% L > 0: initial Lipschitz estimation.
% eta > 1: scaling parameter for backtracking.
% maxits: number of iterations to run prox grad alg.
% tol: stopping tolerance for prox grad algorithm.
%
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Output
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% x: solution at termination.
% k: number of iterations performed.
% L: step length/approximate Lipschitz constant.

%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Initialization.
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Initial solution x.
x = x0;

% Get number of components of x,d, row/cols of A.
n = length(x);

%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Outer loop: Repeat until converged or max # of iterations reached.
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
for k = 0:maxits
    
    %+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    % Check for convergence.
    %+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
      
    % Compute gradient of differentiable part (f(x) = 0.5*x'*A*x - d'*x)
    df = A*x - d;
    
    %----------------------------------------------------------------
    % Compute disagreement between df and lam*sign(x) on supp(x).
    %----------------------------------------------------------------
    % Initialize error vector.
    err = zeros;
    % Initialize cardinality of support.
    card = 0;
   
    % For each i, update error if i in the support.
    for i=1:n
        if abs(x(i)) > 1e-12    % i in supp(x).
            % update cardinality.
            card = card + 1; 
            
            % update error vector.
            err(i) = -df(i) - lam*sign(x(i));
        end
    end
    
    
    %----------------------------------------------------------------
    % Print optimality condition violation.
    %----------------------------------------------------------------    
%     if (k <=2 || mod(k,10) == 0)
%         fprintf('it = %g   inf(df) - lam = %3.2e   inf(err) = %3.2e f = %3.2e\n', k, (norm(df, inf) - lam)/n, norm(err, inf)/n, 0.5*x'*A*x - d'*x + lam*norm(x,1))
%     end
    
    %----------------------------------------------------------------
    % Check stopping criteria -df(x) in subdiff g(x).
    %   Need inf(df) < lam + tol, inf(err) < tol.
    %----------------------------------------------------------------
    if max(norm(df, inf) - lam, norm(err, inf)) < tol*n
        % CONVERGED!!!
%         fprintf('Subproblem converged after %g iterations\n\n\n', k);
        
        break
    else
        % Update x using soft-thresholding and backtracking.      
        alpha = 1/L; % update step length.
        pL = sign(x - alpha*df).*max(abs(x - alpha*df) - lam*alpha*ones(n,1), zeros(n,1)); % initial update.
%         gap = 1/2*(pL - x)'*(L*eye(n) - A)*(pL-x);
        gap = 1/2*(L*norm(pL-x, 2)^2 - (pL-x)'*A*(pL-x));
        % backtrack.
        while (gap < -tol) % Stop when Q - F >= 0.
            L = eta*L; % Scale L by eta.
            alpha = 1/L; % update step length.
            pL = sign(x - alpha*df).*max(abs(x - alpha*df) - lam*alpha*ones(n,1), zeros(n,1)); % update iterate.
%             gap = 1/2*(pL - x)'*(L*eye(n) - A)*(pL-x);
            gap = 1/2*(L*norm(pL-x, 2)^2 - (pL-x)'*A*(pL-x));
        end
        
        
        
        % update iterate.
        x = pL;
        
        
        
    end
    
    
    
    
    
end