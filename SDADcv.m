function [B, Q, lbest, lambest,scores] = SDADcv(train, folds, Om, gam, lams, mu, q, PGsteps, PGtol, maxits, tol, feat, quiet)

% Applies alternating direction method of multipliers with cross validation
% to the optimal scoring formulation of
% sparse discriminant analysis proposed by Clemmensen et al. 2011.
%
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Input
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% train.X, train.Y: training data.
% folds: number of folds to use in K-fold cross-validation.
% Om: p by p parameter matrix Omega in generalized elastic net penalty.
% gam > 0: regularization parameter for elastic net penalty.
% lams > 0: vector of regularization parameters for l1 penalty.
% mu > 0: penalty parameter for augmented Lagrangian term.
% q: desired number of discriminant vectors.
% PGsteps: max its of inner prox-grad algorithm to update beta.
% maxits: number of iterations to run alternating direction alg.
% tol: stopping tolerance for alternating direction algorithm.
% feat: maximum fraction of nonzero features desired in validation scheme.
% quiet: if 1 does not display intermediate stats.
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Output
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% B: p by q by nlam matrix of discriminant vectors.
% Q: K by q by nlam matrix of scoring vectors.
% best_ind: index of best solution in [B,Q].

%% Initialize training sets, etc.

% Extract X and Y from train.
X = train.X;
Y = train.Y;

% Get dimensions of input matrices.
[n, p] = size(X);
[~, K] = size(Y);

% If n is not divisible by K, duplicate some records for the sake of
% cross validation.
% Initialize number of padding observations.
if mod(n,folds) > 0
    % number of elements to duplicate.
    pad = ceil(n/folds)*folds - n;

    % duplicate elements of X and Y.
    X = [X; X(1:pad, :)];
    Y = [Y; Y(1:pad, :)];
end

% Get new size of X.
[n, ~] = size(X);

% Randomly permute rows of X.
prm = randperm(n);
X = X(prm, :);
Y = Y(prm, :);

% Extract class labels.
[~, labs] = max(Y, [],2 );

% Sort lambdas in ascending order (break ties by using largest lambda =
% sparsest vector).
lams = sort(lams, 'ascend');


%% Initialize cross-validation indices.

% Number of validation samples.
nv = n/folds;

% Initial validation indices.
vinds = (1:nv)';

% Initial training indices.
tinds = ((nv+1):n)';

% Get number of parameters to test.
nlam = length(lams);

% Validation scores.
scores = q*p*ones(folds, nlam);

% Misclassification rate for each classifier.
mc = zeros(folds, nlam);

for f = 1 : folds

    %% Initialization.
    % Extract X and Y from train.
    Xt = X(tinds, :);
    Yt = Y(tinds, :);
    [Xt, mut, sigt, ft] = normalize(Xt);

    % Extract training data.
    Xv = X(vinds, :);
%     Yv = Y(vinds, :);
    Xv = normalize_test(Xv, mut, sigt, ft);
    
    % Get dimensions of training matrices.
    [nt, p] = size(Xt);

    % Centroid matrix of training data.
    C = diag(1./diag(Yt'*Yt))*Yt'*Xt;


    % Check if Om is diagonal. If so, use matrix inversion lemma in linear
    % system solves.
    if norm(diag(diag(Om)) - Om, 'fro') < 1e-15

        % Flag to use Sherman-Morrison-Woodbury to translate to
        % smaller dimensional linear system solves.
        SMW = 1;

        % Easy to invert diagonal part of Elastic net coefficient matrix.
        M = mu + 2*gam*diag(Om); 
        Minv = 1./M;

        % Cholesky factorization for smaller linear system.
        RS = chol(eye(nt) + 2*Xt*((Minv/nt).*Xt') );
   

    else % Use Cholesky for solving linear systems in ADMM step.

        % Flag to not use SMW.
        SMW = 0;
        A = mu*eye(p) + 2*(Xt'*Xt/nt + gam*Om); % Elastic net coefficient matrix.
        R2 = chol(A); % Cholesky factorization of mu*I + A.
    end

    %+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    % Matrices for theta update.
    %+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    D = 1/nt*(Yt'*Yt); %D
    %M = X'*Y; % X'Y.
    R = chol(D); % Cholesky factorization of D.

    %% Validation Loop.

    if (quiet == 0)
        fprintf('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
        fprintf('Fold %d \n', f)
        fprintf('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
    end
    %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    % Loop through potential regularization parameters.
    %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    % Initialize B.    
    B = zeros(p, q, nlam);
    
    for ll = 1:nlam

        % Initialize Q.
        Q = ones(K,q);        

        %%
        %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        % Call Alternating Direction Method to solve SDA.
        %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        % For j=1,2,..., q compute the SDA pair (theta_j, beta_j).
        %[f, lams(ll)]
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
            %theta = Mj(theta0);
            theta = theta/sqrt(theta'*D*theta);

            % Initialize coefficient vector for elastic net step.
            d = 2*Xt'*(Yt*(theta/nt));

            % Initialize beta.
            if SMW == 1
                btmp = Xt*(Minv.*d)/nt;
                beta = (Minv.*d) - 2*Minv.*(Xt'*(RS\(RS'\btmp)));
            else
                beta = R2\(R2'\d);
            end

            %+++++++++++++++++++++++++++++++++++++++++++++++++++++
            % Alternating direction method to update (theta, beta)
            %+++++++++++++++++++++++++++++++++++++++++++++++++++++

            for its = 1:maxits

                % Update beta using alternating direction method of multipliers.
                b_old = beta;

                if SMW == 1
                    % Use SMW-based ADMM.
                    [~,beta,~,~] = ADMM_EN_SMW(Minv, Xt, RS, d, beta, lams(ll), mu, PGsteps, PGtol, 1);
                else
                    % Use vanilla ADMM.
                    [~, beta,~, ~] = ADMM_EN2(R2, d, beta, lams(ll), mu, PGsteps, PGtol, 1);
                end

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

                % Check convergence.
                if max(db, dt) < tol
                    % Converged.
                    %fprintf('Algorithm converged for %g iterations\n', j)
                    break
                end
            end %its

            % Update Q and B.
            Q(:,j) = theta;
            B(:,j, ll) = beta;
        end %j.

        %%
        %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        % Get classification statistics for (Q,B).
        %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        % Get prediction/scores.
        stats = predict(B(:,:,ll), [labs(vinds), Xv], C');
        mc(f, ll) = stats.mc;
        
        %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        if  (1<= stats.l0) && (stats.l0 <= q*p*feat) 
        
%         if  (1<= stats.l0 <= q*p*feat)% if fraction nonzero features less than feat.
%             fprintf('Sparse enough. Use MC as score. \n')
            % Use misclassification rate as validation score.
            scores(f, ll) = mc(f, ll);
            %         elseif nnz(B) < 0.5; % Found trivial solution.
            %             %fprintf('dq \n')
            %             scores(f, 11) = 10000; % Disqualify with maximum possible score.
        elseif (stats.l0 > q*p*feat) % Solution is not sparse enough, use most sparse as measure of quality instead.
%             fprintf('Not sparse enough. Use cardinality as score. \n')
            
            scores(f, ll) = stats.l0;
        end


        % Display iteration stats.
        if (quiet ==0)
            fprintf('ll: %3d | lam: %1.5e| feat: %d | mc: %1.5e | score: %1.5e \n',  ll, lams(ll), stats.l0, mc(f, ll), scores(f, ll))
        end


    end % For ll = 1:nlam.

    %+++++++++++++++++++++++++++++++++++
    % Update training/validation split.
    %+++++++++++++++++++++++++++++++++++
    % Extract next validation indices.
    tmp = tinds(1:nv);

    % Update training indices.
    tinds = [tinds((nv+1):nt); vinds];


    % Update validation indices.
    vinds = tmp;

end % folds.

%%  Find best solution.

if quiet == false
    fprintf('Finished training lambda.\n')
end
% average CV scores.
avg_score = mean(scores);

% choose lambda with best average score (break ties by taking largest ->
% most sparse discriminant vector).
minscore = min(avg_score);
lbest = find(avg_score == minscore, 1, 'last');
lambest = lams(lbest);

%% Solve with lambda = lam(lbest).

% Finished training lambda.
fprintf('Finished Training: lam = %d \n', lambest)

% Use full training set.
Xt = train.X;
Yt = train.Y;
Xt = normalize(Xt);

% Solve for B & Q.
[B, Q] = SDAD(Xt, Yt, Om, gam, lambest, mu, q, PGsteps, PGtol, maxits, tol, 1);



