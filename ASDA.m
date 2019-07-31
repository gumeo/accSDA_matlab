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
%   .insteps - positive integer: number of iterations to perform in inner loop.
%   .outsteps - positive integer: number of iterations to perform in outer BCD loop.
%   .intol, .outtol > 0: inner and outer loop stopping tolerances.
%   .folds - positive integer: if cv = true, the number of folds to use.
%   .bt - logical: indicates to use backtracking line search if true, o/w
%       uses constant step size. Only needed for PG/APG.
%   .L: if bt true, the initial value of possible Lipschitz constant.
%   .eta >0: scaling factor in backtracking line search.
%   .feat - in [0,1]: if cv true, the desired max cardinality of 
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% OUTPUT.
% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% B: p by (K-1)  matrix of discriminant vectors.
% Q: K by (K-1)  matrix of scoring vectors.

% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% PROXIMAL GRADIENT.
% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if method == "PG"  % Proximal gradient
    if cv == false % No cross validation.
        
        
    end % if cv.
else % Method not allowed.
    error('Not a valid method. Please choose from "PG", "APG", or "ADMM".')
end % if method.
    



end

