function [B, Q] = ASDA(X, Y, Om, gam, lam, cv, method, opts)
% ASDA Block coordinate descent for sparse optimal scoring.
% 
% Applies accelerated proximal gradient algorithm 
% to the optimal scoring formulation of
% sparse discriminant analysis proposed by Clemmensen et al. 2011.
% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% INPUT.
% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% X: n by p data matrix.
% Y: n by K matrix of indicator variables (Yij = 1 if i in classs j)
% Om: p by p parameter matrix Omega in generalized elastic net penalty.
% gam > 0: regularization parameter for ridge penalty
% lam > 0: regularization parameter(s) for l1 penalty.
%   If cv = true, then this is a list of possible values.
%   Otherwise, a single value for the regularization parameter.
% cv - logical: flag whether to use cross-validation.
% method: indicate which method to use to update beta variable. Choose from: 
%   "PG" - proximal gradient method.
%   "APG" - accelerate proximal gradient method.
%   "ADMM" - alternating direction method of multipliers.
% opts: additional arguments needed by each method.
%   .q - integer between 1 and K-1: number of discriminant vectors to
%       calculate.
%   .insteps - positive integer: number of iterations to perform in inner loop.
%   .outsteps - positive integer: number of iterations to perform in outer BCD loop.
%   .intol, .outtol > 0: inner and outer loop stopping tolerances.
%   .folds - positive integer: if cv = true, the number of folds to use.
%   .bt - logical: indicates to use backtracking line search if true, o/w
%       uses constant step size. Only needed for PG/APG.
%   .L: if bt true, the initial value of possible Lipschitz constant.
%   .eta >0: scaling factor in backtracking line search.
%   .feat - in [0,1]: if cv true, the desired max cardinality of dvs.
%   .quiet - logical: indicate whether to display intermediate stats.
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% OUTPUT.
% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% B: p by (K-1)  matrix of discriminant vectors.
% Q: K by (K-1)  matrix of scoring vectors.

% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% PRELIMINARY ERROR CHECKING AND INITIALIZATION.
% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

% Get data size.
[n,p] = size(X);
[n1, K] = size(Y);

% Check that Y and X have same number of observations.
if n~=n1
    error('X and Y must contain same number of rows!')
end

% Check if q is correct form.
q = opts.q;
if  floor(q)~= q || q < 1 || q > (K-1)
    error('q must be a integer between 1 and K-1.')
end

% Check if steps is in correct form.
insteps = opts.insteps;
outsteps = opts.outsteps;
if floor(insteps)~=insteps || insteps < 1
    error('Inner steps must be a positive integer')
end

if floor(outsteps)~=outsteps || outsteps < 1
   error('Outer steps must be a positive integer')
end 

% Check if tolerances are in correct form.
intol = opts.intol;
if intol <= 0
    error('Subproblem stopping tolerance must be positive.')
end
outtol = opts.outtol;
if outtol <= 0
    error('Stopping tolerance must be positive.')
end

% Check if gamma is positive.
if gam < 0 
    error('Gamma must be nonnegative');
end

% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% PROXIMAL GRADIENT.
% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if method == "PG"  % Proximal gradient        
    
    if cv == false % No cross validation.
        
        % Check lambda.
        if length(lam) > 1 || lam < 0
            error('Lambda must be a positive integer if not using CV.')
        end
        
        %+++++++++++++++++++++++++++++++++++
        % PG, no CV, no BT.
        %+++++++++++++++++++++++++++++++++++
        if opts.bt == false            
            fprintf('Calling proximal gradient with constant step size. \n')
            % Call SDAP.
            [B,Q] = SDAP(X,Y, Om, gam, lam, q, insteps, intol, outsteps, outtol, opts.quiet);
        
        %+++++++++++++++++++++++++++++++++++
        % PGB, no CV
        %+++++++++++++++++++++++++++++++++++
        elseif opts.bt == true % PGB, no CV.            
            fprintf('Calling proximal gradient with backtracking line search. \n')            
            
            % Check input.
            if opts.L <=0
                error('Initial Lipschitz constant estimate must be positive.')
            end
            
            if opts.eta <= 1
                error('Backtracking scaling factor must be > 1.')
            end
            
            % Call SDAPbt.
            [B,Q] = SDAPbt(X, Y, Om, gam, lam, opts.L, opts.eta, q, insteps, intol, outsteps, outtol, opts.quiet); 
            
        else % opts.bt missing or not logical.
            error('opts.bt must be logical if using proximal gradient method.')            
        
        end % PG, no CV              
        
    end % if cv.
    
% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% ACCELERATED PROXIMAL GRADIENT.
% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
elseif method == "APG" % Use accelerated proximal gradient.
    
    if cv == false % No cross validation.
        
        % Check lambda.
        if length(lam) > 1 || lam < 0
            error('Lambda must be a positive integer if not using CV.')
        end
        
        %+++++++++++++++++++++++++++++++++++
        % PG, no CV, no BT.
        %+++++++++++++++++++++++++++++++++++
        if opts.bt == false            
            fprintf('Calling proximal gradient with constant step size. \n')
            % Call SDAP.
            [B,Q] = SDAP(X,Y, Om, gam, lam, q, insteps, intol, outsteps, outtol, opts.quiet);
        
        %+++++++++++++++++++++++++++++++++++
        % PGB, no CV
        %+++++++++++++++++++++++++++++++++++
        elseif opts.bt == true % PGB, no CV.            
            fprintf('Calling proximal gradient with backtracking line search. \n')            
            
            % Check input.
            if opts.L <=0
                error('Initial Lipschitz constant estimate must be positive.')
            end
            
            if opts.eta <= 1
                error('Backtracking scaling factor must be > 1.')
            end
            
            % Call SDAPbt.
            [B,Q] = SDAPbt(X, Y, Om, gam, lam, opts.L, opts.eta, q, insteps, intol, outsteps, outtol, opts.quiet); 
            
        else % opts.bt missing or not logical.
            error('opts.bt must be logical if using proximal gradient method.')            
        
        end % PG, no CV              
        
    end % if cv.
    
else % Method not allowed.
    error('Not a valid method. Please choose from "PG", "APG", or "ADMM".')
end % if method.
    



end
