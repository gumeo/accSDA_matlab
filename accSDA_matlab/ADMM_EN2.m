function [x,y,z, k] = ADMM_EN2(R, d, x0, lam, mu, maxits, tol, quiet)
% ADMM_EN2 admm for SOS elastic net problem using SMW lemma.
% Applies Alternating Direction Method of Multipliers to the l1-regularized
% quadratic program
%   f(x) + g(x) = 0.5*x'*A*x - d'*x + lam*l1(x).
%
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% INPUT.
% A: n by n positive definite coefficient matrix
% R: upper triangular matrix in Chol decomp muI + A = R'*R;
% d: n dim coefficient vector.
% lam > 0: regularization parameter for l1 penalty.
% mu > 0: augmented Lagrangian penalty parameter.
% alpha: step length.
% maxits: number of iterations to run.
% tol = [tol.abs, tol.rel]: stopping tolerances.
% quiet = control display of intermediate statistics.
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% OUTPUT.
% (x, y,z): solution at termination.
% k: number of iterations needed.

%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Initialization.
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Initial solutions.
x = x0;
y = x0;
p = length(x);
z = zeros(p,1);


%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Outer loop: Repeat until converged or max # of iterations reached.
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if quiet == false
    fprintf('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
    fprintf('InIt \t\t + pgap \t\t + dgap \t\t + norm(dvs) \n')
    fprintf('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
end
for k = 0:maxits
    
    %+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    % Update x.
    %+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    % (mu I + A)x = d + mu*y - z.
    b = d + mu*y-z;
    Rx = R'\b;
    x = R\Rx;
    
    %norm(x, 'inf')
    
    %+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    % Update y.
    %+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    % Update y using soft-thresholding.
    yold = y;
    tmp = x + z/mu;
    y = sign(tmp).*max(abs(tmp) - lam/mu*ones(p,1), zeros(p,1));
    
    %+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    % Update z.
    %+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    z = z + mu*(x-y);
    
    %+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    % Check for convergence.
    %+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    %%% Primal constraint violation.
    
    % Primal residual.
    r = x - y;
    
    % l2 norm of the residual.
    dr = norm(r);
    
    %%% Dual constraint violation.
    
    % Dual residual.
    s = mu*(y - yold);
    
    % l2 norm of the residual.
    ds = norm(s);
    
    %%%  Check if the stopping criteria are satisfied.
    
    % Compute absolute and relative tolerances.
    ep = sqrt(p)*tol.abs + tol.rel*max(norm(x), norm(y));
    es = sqrt(p)*tol.abs + tol.rel*norm(y);
    
    % Display current iteration stats.
    if (k <=2 || mod(k,10) == 0) && quiet==false
        fprintf('%3g \t\t + %1.2e \t +  %1.2e \t + %1.2e \n', k, dr-ep, ds-es, max(norm(x), norm(y)))
    end
    
    
    % Check if the residual norms are less than the given tolerance.
    if (dr < ep && ds < es)
         % CONVERGED!!!
        if quiet==false
            fprintf('Subproblem converged after %g iterations\n', k)
        end
        break % The algorithm has converged.
    end
end
