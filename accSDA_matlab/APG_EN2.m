function [x, k] = APG_EN2(A, d, x0, lam, alpha,  maxits, tol, quiet)
% APG_EN2 - accelerated proximal gradient method for SOS problem.
% Applies accelerated proximal gradient algorithm to the l1-regularized quad
%   f(x) + g(x) = 0.5*x'*A*x - d'*x + lam*l1(x).
%
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% INPUT.
% A: p by p positive definite coefficient matrix 
%       A = 2(gamma*Om + X'X/n).
% d: p dim coefficient vector.
% lam > 0: regularization parameter for l1 penalty.
% alpha: step length.
% maxits: number of iterations to run prox grad alg.
% tol: stopping tolerance for prox grad algorithm.
% quiet: toggle display of intermediate output.
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% OUTPUT.
% x: solution at termination.
% k: number of iterations performed.

%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Initialization.
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Initial solution x.
x = x0;
xold = x;



% Get number of components of x,d, row/cols of A.
p = length(x);

% Initial momentum coefficient.
t = 1;
told = 1;

% Objective function and gradient.
if A.flag == 1     
    df = @(x) 2*(A.gom.*x + A.X'*(A.X*(x/A.n))) - d;
else
    df = @(x) A.A*x - d;
end

%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Outer loop: Repeat until converged or max # of iterations reached.
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if quiet == false
    fprintf('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
    fprintf('InIt \t + inf(df) - lam \t + inf(err) \n')
    fprintf('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
end
for k = 0:maxits
    
    
    %+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    % Check for convergence.
    %+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    % Compute gradient of differentiable part (f(x) = 0.5*x'*A*x - d'*x)
    dfx = df(x);
    
    
    %----------------------------------------------------------------
    % Compute disagreement between df and lam*sign(x) on supp(x).
    %----------------------------------------------------------------
    % Initialize error vector.
    err = zeros(p,1);
    % Initialize cardinality of support.
    card = 0;
   
    % For each i, update error if i in the support.
    for i=1:p
        if abs(x(i)) > 1e-12    % i in supp(x).
            % update cardinality.
            card = card + 1; 
            
            % update error vector.
            err(i) = -dfx(i) - lam*sign(x(i));
        end
    end
    
    %----------------------------------------------------------------
    % Print optimality condition violation.
    %----------------------------------------------------------------    
     if (k <=2 || mod(k,10) == 0) && quiet==false
      fprintf('%3g \t +  %1.2e \t\t +  %1.2e \n', k, (norm(dfx, inf) - lam)/p, norm(err, inf)/p)
    end
    
    %----------------------------------------------------------------
    % Check stopping criteria -df(x) in subdiff g(x).
    %   Need inf(df) < lam + tol, inf(err) < tol.
    %----------------------------------------------------------------
    if max(norm(dfx, inf) - lam, norm(err, inf)) < tol*p
        % CONVERGED!!!
        if quiet==false
            fprintf('Subproblem converged after %g iterations\n', k)
        end

        break
    else
        %------------------------------------------------------------
        % Update x APG iteration.
        %------------------------------------------------------------
               
        
        % Compute extrapolation factor.
        told = t;
        t = (1 + sqrt(1 + 4*told^2))/2;        
        
        % Extrapolate using last two iterates.
        y = x + (told - 1)/t*(x - xold);
        
        % Compute value and gradient at y of f at y.
        dfy = df(y);
               
        % Take proximal gradient step from y.
        xold = x;
        x = sign(y - alpha*dfy).*max(abs(y - alpha*dfy) - lam*alpha*ones(p,1), zeros(p,1));
               
    end
        
end
    
    
end