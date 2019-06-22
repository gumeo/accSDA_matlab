
function [x,y,z,its, errtol] = SZVD_ADMM(B,  N, D, sols0, pen_scal, gamma, beta, tol, maxits, quiet)

% Iteratively solves the problem
%       min{-1/2*x'B'x + gamma p(y): l2(x) <= 1, DNx = y}
% using ADMM.
%====================================================================
% Input.
%====================================================================
%   B: Between class covariance matrix for objective (in space defined by N).
%   N: basis matrix for null space of covariance matrix W.
%   D: penalty dictionary/basis.
%   sols0: initial solutions sols0.x, sols0.y, sols0.z
%   pen_scal: penalty scaling term.
%   gamma:  l1 regularization parameter.
%   beta:    penalty term controlling the splitting constraint.
%   tol:    tol.abs = absolute error, tol. rel = relative error to be
%                   achieved to declare convergence of the algorithm.
%   maxits: maximum number of iterations of the algorithm to run.
%   quiet: toggles between displaying intermediate statistics.
%====================================================================
% Output:.
%====================================================================
%   x, y, z: iterates at termination.
%   its: number of iterations required to converge.
%   errtol: stopping error bound at termination.


%====================================================================
% Precomputes quantities that will be used repeatedly by the algorithm.
%====================================================================

% Dimension of decision variables.
p = size(D, 1);

% Compute D*N.
if D == eye(p);
    DN = N;
else
    DN = D*N;
end


%====================================================================
% Initialize x solution and constants for x update.
%====================================================================
if (size(B,2) == 1) %% K = 2 case.
    
    % Compute (DN)'*(mu1-mu2)
    w = DN' * B    ;
    
    % Compute initial x.
    x = sols0.x;    
    
    % constant for x update step.
    Xup = beta - w'*w;
    Xup = Xup(1,1);
    
    
else    %% K > 2 Case.
    
    % Dimension of the null-space of W.
    l= size(N, 2);
    
    % Compute initial x.
    x = sols0.x;
    
    
    % Take Cholesky of beta I - B (for use in update of x)
    L = chol(beta*eye(l) - B, 'lower');
end

%====================================================================
% Initialize decision variables y and z.
%====================================================================

% Initialize y and z.
y = sols0.y;
z = sols0.z;



%====================================================================
%% Call the algorithm.
%====================================================================

for iter=1:maxits
    
    
    %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    % Step 1: Perform shrinkage to update y_k+1.
    %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    % Save previous iterate.
    yold = y;
    
    % Call soft-thresholding.       
    y = vec_shrink(beta*DN * x + z, gamma * pen_scal);
    
    % Normalize y (if necessary).
    tmp = max(0, norm(y) - beta);
    y = y/(beta + tmp);
    
    % Truncate complex part (if have +0i terms appearing.)
    y = real(y);
    
    %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    % Step 2: Update x_k+1 by solving
    % x_k+1 = argmin { -x'*A*x + beta/2 l2(x - y_k+1 + z_k)^2}
    %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    % Compute RHS.
    b = DN'*(beta*y - z);
    
    if (size(B,2)==1)    %% K=2
        
        % Update using Sherman-Morrison formula.
        x = (b + (b' * w)*w/Xup)/beta;
        
    else  %% K > 2.
        
        % Update using by solving the system LL'x = b.
        btmp = L\b;
        x = L'\btmp;
%        x = (beta*eye(l) - B)\b;
    end %if.
    
    % Truncate complex part (if have +0i terms appearing.)
    x = real(x);
    
    %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    %  Step 3: Update the Lagrange multipliers
    % (according to the formula z_k+1 = z_k + beta*(N*x_k+1 - y_k+1) ).
    %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    %zold = z;
    z = real(z + beta*(DN*x - y));
    
    %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    % Check stopping criteria.
    %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    %----------------------------------------------------------------
    % Primal constraint violation.
    %----------------------------------------------------------------
    % Primal residual.
    r = DN*x - y;    
    % l2 norm of the residual.
    dr = norm(r);
	
	%----------------------------------------------------------------   
    % Dual constraint violation.
    %----------------------------------------------------------------
    % Dual residual.
    s = beta*(y - yold);    
    % l2 norm of the residual.
    ds = norm(s);
    
	%----------------------------------------------------------------
    % Check if the stopping criteria are satisfied.
	%----------------------------------------------------------------
    
    % Compute absolute and relative tolerances.
    ep = sqrt(p)*tol.abs + tol.rel*max(norm(x), norm(y));
    es = sqrt(p)*tol.abs + tol.rel*norm(y);
    
    % Display current iteration stats.
    if (quiet==0 & mod(iter, 5) == 0)
        fprintf('it = %g, primal_viol = %3.2e, dual_viol = %3.2e, norm_DV = %3.2e\n', iter, dr-ep, ds-es, norm(y))
    end
    
    % Check if the residual norms are less than the given tolerance.
    if (dr < ep && ds < es && iter > 10)
        break % The algorithm has converged.
    end
    
end %for.


%====================================================================
% Output results.
%====================================================================

if maxits > 0
    its = iter;
    errtol = min(ep,es);
else
    its = 0;
    errtol=0;
end

end






