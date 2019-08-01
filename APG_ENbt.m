function [x, k,L] = APG_ENbt(A, d, x0, lam, L, eta,  maxits, tol, quiet)
% APG_ENBT apg method with backtracking for SOS problem.
% Applies accelerated proximal gradient algorithm to the l1-regularized quad
%   f(x) + g(x) = 0.5*x'*A*x - d'*x + lam*l1(x).
% Uses back tracking line search.
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% INPUT.
% A: p by p positive definite coefficient matrix 
%       A = 2(gamma*Om + X'X/n).
% d: p dim coefficient vector.
% lam > 0: regularization parameter for l1 penalty.
% L: initial value of backtracking Lipschitz constant.
% eta: backtracking scaling parameter.
% maxits: number of iterations to run prox grad alg.
% tol: stopping tolerance for prox grad algorithm.
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% OUTPUT.
% x: solution at termination.
% k: number of iterations performed.
% L: approximate Lipschitz constant returned by line-search.

%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Initialization.
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Initial solution x.
x = x0;
%xold = x;
y = x;


% Get number of components of x,d, row/cols of A.
p = length(x);

% Initial momentum coefficient.
t = 1;
alph = 1;
%told = 1;

% Objective function and gradient.
if A.flag == 1 
    f = @(x) x'*(A.gom.*x) + 1/A.n*norm(A.X*x)^2 + d'*x;
    df = @(x) 2*(A.gom.*x + A.X'*(A.X*(x/A.n))) - d;
else
    f = @(x) 0.5*x'*A.A*x + d'*x;
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
        % Backtracking line search update.
        %------------------------------------------------------------
        % Step length.
        alph = 1/L;
        % Evaluate proximal gradient at y_k.
        dfy = df(y);
        pLy = sign(y - alph*dfy).*max(abs(y - alph*dfy) - lam*alph*ones(p,1), zeros(p,1));
        % Q - F, backtrack if negative.
        dy = pLy - y;
        
        if A.flag == 1
            QminusF = 1/2*L*norm(dy,2)^2 - 1/A.n*norm(A.X*dy)^2 - dy'*((A.gom).*dy);
        else
            QminusF =  1/2*(L*norm(dy, 2)^2 - dy'*A.A*dy);
        end
        
%         diffold = 1/2*(L*norm(dy, 2)^2 - dy'*A.A*dy) - QminusF;
%         fprintf('Q-F: %1.3e \n', QminusF)
%         fprintf('diff: %1.3e \n', diffold)
%         fprintf('diffA: %1.3e \n', dy'*A.A*dy - 2*1/A.n*norm(A.X*dy)^2 - 2*dy'*((A.gom).*dy));
        
%         QminusF = 1/2*(L*norm(dy, 2)^2 - x'*(A.gom.*x) + 1/A.n*norm(A.X*x)^2
        
        % update Q - F.
        
        while (QminusF < -tol)
            % Update approximate Lipschitz constant.
            L = eta*L;
            % Update step length.
            alph = 1/L;
            % Update proximal gradient iterate.
            pLy = sign(y - alph*dfy).*max(abs(y - alph*dfy) - lam*alph*ones(p,1), zeros(p,1));
            
            % Update gap statistic.
            dy = pLy - y;
            if A.flag == 1
                QminusF = 1/2*L*norm(dy,2)^2 - 1/A.n*norm(A.X*dy)^2 - dy'*((A.gom).*dy);
            else
                QminusF =  1/2*(L*norm(dy, 2)^2 - dy'*A.A*dy);
            end
           %  fprintf('Q-F: %1.3e \n', QminusF)
%             QminusF = 1/2*(L*norm(dy, 2)^2 - dy'*A.A*dy);
  
        end
        
        
        % Update x by APG iteration.
        xold = x;
        x = pLy;
        
        % Compute extrapolation factor.
        told = t;
        t = (1 + sqrt(1 + 4*told^2))/2;
        
        % Extrapolate using last two iterates.
        y = x + (told - 1)/t*(x - xold);        
    

    end
        
end
    
    
end