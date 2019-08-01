function ASDAres = ASDA(X, Y, Om, gam, lam, cv, method, q, insteps, outsteps, intol, outtol, quiet, opts)
% ASDA Block coordinate descent for sparse optimal scoring.
%
% Applies accelerated proximal gradient algorithm
% to the optimal scoring formulation of
% sparse discriminant analysis proposed by Clemmensen et al. 2011.
% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% INPUT.
% X: n by p data matrix.
% Y: n by K matrix of indicator variables (Yij = 1 if i in classs j)
% Om: p by p parameter matrix Omega in generalized elastic net penalty.
% gam > 0: regularization parameter for ridge penalty
% lam > 0: regularization parameter(s) for l1 penalty.
%   If cv = true, then this is a list of possible values.
%   Otherwise, a single value for the regularization parameter.

% cv - logical: flag whether to use cross-validation.
% method: indicate which method to use to update beta variable. Choose from:
% q - integer between 1 and K-1: number of discriminant vectors to
%       calculate.
% insteps - positive integer: number of iterations to perform in inner loop.
% outsteps - positive integer: number of iterations to perform in outer BCD loop.
% intol, outtol > 0: inner and outer loop stopping tolerances.
% quiet - logical: indicate whether to display intermediate stats.
% method: solver for updating discriminant vectors.
%   "PG" - proximal gradient method.
%   "APG" - accelerate proximal gradient method.
%   "ADMM" - alternating direction method of multipliers.
% opts: additional optional arguments needed by each method.
%   .folds - positive integer: if cv = true, the number of folds to use.
%   .bt - logical: indicates to use backtracking line search if true, o/w
%       uses constant step size. Only needed for PG/APG.
%   .L: if bt true, the initial value of possible Lipschitz constant.
%   .eta > 0: scaling factor in backtracking line search.
%   .mu > 0: augmented Lagrangian penalty parameter for ADMM.
%   .feat - in [0,1]: if cv true, the desired max cardinality of dvs.
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% OUTPUT.
% ASDAres - structure containing:
% B: p by q  matrix of discriminant vectors.
% Q: K by q  matrix of scoring vectors.
% bestind: index of best regularization parameter (if using CV).
% bestlam: best regularization parameter (if using CV).
% cvscores: matrix of cross validation scores (if using CV). 

% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% PRELIMINARY ERROR CHECKING AND INITIALIZATION.
% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

% Get data size.
[n,~] = size(X);
[n1, K] = size(Y);

% Check that Y and X have same number of observations.
if n~=n1
    error('X and Y must contain same number of rows!')
end

% Check if q is correct form.
if  floor(q)~= q || q < 1 || q > (K-1)
    error('q must be a integer between 1 and K-1.')
end

% Check if steps is in correct form.
if floor(insteps)~=insteps || insteps < 1
    error('Inner steps must be a positive integer')
end

if floor(outsteps)~=outsteps || outsteps < 1
   error('Outer steps must be a positive integer')
end

% Check if tolerances are in correct form.
if intol <= 0
    error('Subproblem stopping tolerance must be positive.')
end

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
            fprintf('Proximal gradient with constant step size (PG). \n')
            % Call SDAP.
            [B,Q] = SDAP(X,Y, Om, gam, lam, q, insteps, intol, outsteps, outtol, quiet);

        %+++++++++++++++++++++++++++++++++++
        % PGB, no CV
        %+++++++++++++++++++++++++++++++++++
        elseif opts.bt == true % PGB, no CV.
            fprintf('Proximal gradient with backtracking line search (PGB). \n')

            % Check input.
            if opts.L <=0
                error('Initial Lipschitz constant estimate must be positive.')
            end

            if opts.eta <= 1
                error('Backtracking scaling factor must be > 1.')
            end

            % Call SDAPbt.
            [B,Q] = SDAPbt(X, Y, Om, gam, lam, opts.L, opts.eta, q, insteps, intol, outsteps, outtol, quiet);

        else % bt missing or not logical.
            error('bt must be logical if using proximal gradient method.')

        end % PG, no CV

    %+++++++++++++++++++++++++++++++++++++++++++++++++++++++
    elseif cv == true % PG CV.
        % Check lam.
        if min(lam) <= 0
            error('Lambda must be a vector of positive real numbers if using CV.')
        end

        % Check feat.
        if opts.feat < 0 || opts.feat > 1
            error('Maximum fraction of nonzero entries must be between 0 and 1.')
        end

        % Check folds.
        if opts.folds < 0 || opts.folds > n || opts.folds ~= floor(opts.folds)
            error('Number of folds must be an integer between 1 and n.')
        end

        % Prepare train.
        train.X = X;
        train.Y = Y;

        %+++++++++++++++++++++++++++++++++++
        % PG, with CV, but no BT.
        %+++++++++++++++++++++++++++++++++++
        if opts.bt == false
            fprintf('Cross validation using proximal gradient with constant step size (PG). \n')

            % Call SDAP.
            [B, Q, lbest, lambest,scores] = SDAPcv(train, opts.folds, Om, gam, lam, q, insteps, intol, outsteps, outtol, opts.feat, quiet);

        %+++++++++++++++++++++++++++++++++++
        % PGB, with CV and BT
        %+++++++++++++++++++++++++++++++++++
        elseif opts.bt == true % PGB, with CV.
            fprintf('Cross validation using proximal gradient with backtracking line search (PGB). \n')

            % Check input.
            if opts.L <=0
                error('Initial Lipschitz constant estimate must be positive.')
            end

            if opts.eta <= 1
                error('Backtracking scaling factor must be > 1.')
            end

            % Call SDAPbt.
            [B, Q, lbest, lambest,scores] = SDAPbtcv(train, opts.folds, Om, gam, lam, opts.L, opts.eta, q, insteps, intol, outsteps, outtol, opts.feat, quiet);

        else % bt missing or not logical.
            error('bt must be logical if using proximal gradient method.')

        end % if BT.

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
        % APG, no CV, no BT.
        %+++++++++++++++++++++++++++++++++++
        if opts.bt == false
            fprintf('Accelerated proximal gradient with constant step size (APG). \n')
            % Call SDAP.
            [B,Q] = SDAAP(X,Y, Om, gam, lam, q, insteps, intol, outsteps, outtol, quiet);

        %+++++++++++++++++++++++++++++++++++
        % APGB, no CV
        %+++++++++++++++++++++++++++++++++++
        elseif opts.bt == true % PGB, no CV.
            fprintf('Accelerated proximal gradient with backtracking line search (APGB). \n')

            % Check input.
            if opts.L <=0
                error('Initial Lipschitz constant estimate must be positive.')
            end

            if opts.eta <= 1
                error('Backtracking scaling factor must be > 1.')
            end

            % Call SDAPbt.
            [B,Q] = SDAAPbt(X, Y, Om, gam, lam, opts.L, opts.eta, q, insteps, intol, outsteps, outtol, quiet);

        else % bt missing or not logical.
            error('bt must be logical if using proximal gradient method.')

        end % if BT.

    %+++++++++++++++++++++++++++++++++++++++++++++++++++++++
    elseif cv == true % APG CV.
        % Check lam.
        if min(lam) <= 0
            error('Lambda must be a vector of positive real numbers if using CV.')
        end

        % Check feat.
        if opts.feat < 0 || opts.feat > 1
            error('Maximum fraction of nonzero entries must be between 0 and 1.')
        end

        % Check folds.
        if opts.folds < 0 || opts.folds > n || opts.folds ~= floor(opts.folds)
            error('Number of folds must be an integer between 1 and n.')
        end

        % Prepare train.
        train.X = X;
        train.Y = Y;

        %+++++++++++++++++++++++++++++++++++
        % APG, with CV, but no BT.
        %+++++++++++++++++++++++++++++++++++
        if opts.bt == false
            fprintf('Cross validation using accelerated proximal gradient with constant step size (APG). \n')

            % Call SDAAP.
            [B, Q, lbest, lambest,scores] = SDAAPcv(train, opts.folds, Om, gam, lam, q, insteps, intol, outsteps, outtol, opts.feat, quiet);

        %+++++++++++++++++++++++++++++++++++
        % APGB, with CV and BT
        %+++++++++++++++++++++++++++++++++++
        elseif opts.bt == true % APGB, with CV.
            fprintf('Cross validation using accelerated proximal gradient with backtracking line search (APGB). \n')

            % Check input.
            if opts.L <=0
                error('Initial Lipschitz constant estimate must be positive.')
            end

            if opts.eta <= 1
                error('Backtracking scaling factor must be > 1.')
            end

            % Call SDAPbt.
            [B, Q, lbest, lambest,scores] = SDAAPbtcv(train, opts.folds, Om, gam, lam, opts.L, opts.eta, q, insteps, intol, outsteps, outtol, opts.feat, quiet);

        else % bt missing or not logical.
            error('bt must be logical if using proximal gradient method.')

        end % if BT.

    else % WRONG INPUT FOR CV.
        error('CV must be logical.')
    end % if cv.

% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% ALTERNATING DIRECTION METHOD OF MULTIPLIERS.
% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
elseif method == "ADMM"
    % Check mu.
    if opts.mu <= 0
        error('Augmented Lagrangian parameter mu must be positive.')
    end

    % Prepare input for stopping tolerances.
    PGtol.abs = intol;
    PGtol.rel = intol;

    %+++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if cv == false % NO CV.

        % Check lambda.
        if length(lam) > 1 || lam < 0
            error('Lambda must be positive if not using CV.')
        end

        % Call ADMM.
        fprintf('Alternating direction method of multipliers (ADMM).\n')
        [B,Q] = SDAD(X, Y, Om, gam, lam, opts.mu, q, insteps, PGtol, outsteps, outtol, quiet);

    %+++++++++++++++++++++++++++++++++++++++++++++++++++++++
    elseif cv == true % ADMM CV.
        fprintf('Cross validation for ADMM. \n')

        % Check lam.
        if min(lam) <= 0
            error('Lambda must be a vector of positive real numbers if using CV.')
        end

        % Check feat.
        if opts.feat < 0 || opts.feat > 1
            error('Maximum fraction of nonzero entries must be between 0 and 1.')
        end

        % Check folds.
        if opts.folds < 0 || opts.folds > n || opts.folds ~= floor(opts.folds)
            error('Number of folds must be an integer between 1 and n.')
        end

        % Prepare train.
        train.X = X;
        train.Y = Y;

        % Call CV with ADMM.
        [B, Q, lbest, lambest,scores]  = SDADcv(train, opts.folds, Om, gam, lam, opts.mu, q, insteps, PGtol, outsteps, outtol, opts.feat, quiet);

    else % WRONG INPUT FOR CV.
        error('CV must be logical.')

    end % if cv.
else % Method not allowed.
    error('Not a valid method. Please choose from "PG", "APG", or "ADMM".')
end % if method.

% Prepare output.
ASDAres.B = B;
ASDAres.Q = Q;
if cv == true
    ASDAres.bestind = lbest;
    ASDAres.bestlam = lambest;
    ASDAres.cvscores = scores;
end


end
