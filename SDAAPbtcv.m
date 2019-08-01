function [B, Q, lbest, lambest,scores] = SDAAPbtcv(train, folds, Om, gam, lams, L, eta, q, PGsteps, PGtol, maxits, tol, feat, quiet)
% SDAAPBTCV apg with backtracking and cross validation for the SOS problem.
% Applies accelerated proximal gradient algorithm with cross validation
% and backtracking line search to the optimal scoring formulation of
% sparse discriminant analysis proposed by Clemmensen et al. 2011.
%
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% INPUT.
% train: training data set.
% folds: number of folds to use in K-fold cross-validation.
% Om: p by p parameter matrix Omega in generalized elastic net penalty.
% gam > 0: regularization parameter for elastic net penalty.
% lams > 0: vector of regularization parameters for l1 penalty.
% L, eta: line-search parameters.
% q: desired number of discriminant vectors.
% PGsteps: max its of inner prox-grad algorithm to update beta.
% maxits: number of iterations to run alternating direction alg.
% tol: stopping tolerance for alternating direction algorithm.
% feat: maximum fraction of nonzero features desired in validation scheme.
% quiet: toggles display of iteration stats.
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% OUTPUT.
% B: p by q by nlam matrix of discriminant vectors.
% Q: K by q by nlam matrix of scoring vectors.
% best_ind: index of best solution in [B,Q].


%% Initialize training sets, etc.

% Sort lambdas in ascending order (break ties by using largest lambda =
% sparsest vector).
lams = sort(lams, 'ascend');

% Extract X and Y from train.
X = train.X;
Y = train.Y;
% [~, labs] = max(Y, [],2 );

% Get dimensions of input matrices.
[n, p] = size(X);
[~, K] = size(Y);

% If n is not divisible by K, duplicate some records for the sake of
% cross validation.
pad = 0; % Initialize number of padding observations.
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
% [~, labs] = max(Y'); labs = labs';

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
    [Xt, mut, sigt,ft] = normalize(Xt);
    Omn = Om(ft, ft);
    Yt = Y(tinds, :);

    % Extract training data.
    Xv = X(vinds, :);
    Xv = normalize_test(Xv, mut, sigt, ft);
    %Yv = Y(vinds, :);
    % Get dimensions of training matrices.
    [nt, p] = size(Xt);

    % Centroid matrix of training data.
    C = diag(1./diag(Yt'*Yt))*Yt'*Xt;


    % Precompute repeatedly used matrix products
    %display('form EN coefficient matrix')
    %gamOm = gam*Om;

    if norm(diag(diag(Omn)) - Omn, 'fro') < 1e-15 % Omega is diagonal.
        A.flag = 1;
        % Store components of A.
        A.gom = gam*diag(Omn);
        A.X = Xt;
        A.n = nt;

       % alpha = 1/( 2*(norm(Xt,1)*norm(Xt,'inf')/nt + norm(A.gom, 'inf') ));
        %alpha = 1/( 2*(norm(X)^2/n + norm(A.gom, 'inf') ));
        %     A.A = 2*(X'*X/n + diag(A.gom));
        %     fprintf('Test bounds: L = %g, Lt = %g', norm(A.A), 2*(norm(X,1)*norm(X,'inf')/n + norm(gamOm, 'inf') ))
        %     norm(A.A)
    else
        A.flag = 0;
        A.A = 2*(Xt'*Xt/nt + gam*Omn); % Elastic net coefficient matrix.
        %alpha = 1/norm(A.A, 'fro');
    end
    
    D = 1/nt*(Yt'*Yt); %D
    R = chol(D);


    %% Validation Loop.

    if (quiet ==0)
        fprintf('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
        fprintf('Fold %d \n', f)
        fprintf('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
    end
    
    % Initialize B and Q.
    %Q = ones(K,q, nlam);
    B = zeros(p, q, nlam);
    
    %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    % Loop through potential regularization parameters.
    %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    for ll = 1:nlam

%         % Initialize B and Q.
        Q = ones(K,q);
%         B = zeros(p, q);

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
            theta = theta/sqrt(theta'*D*theta);

            % Initialize beta.
            if (ll==1) % First lam.                
                if norm(diag(diag(Omn)) - Omn, 'fro') < 1e-15 % Use diagonal initializer.
                    % Extract reciprocal of diagonal of Omega.
                    ominv = 1./diag(Omn);
                    
                    % Compute rhs of f minimizer system.                                        
                    rhs0 = Xt'*(Yt*(theta/nt));
                    size(Xt), size(ominv), size(rhs0)
                    rhs = Xt*((ominv/n).*rhs0);
                    
                    
                    
                    % Compute partial solution.
%                     size(Xt)
%                     nt
%                     size(eye(nt))
%                     size((Xt*((ominv/(gam*n)).*Xt')))
%                     size(rhs)
%                     
                    tmp = (eye(nt) + Xt*((ominv/(gam*nt)).*Xt'))\rhs;
                    
                    % Finishing solving for beta using SMW.
                    beta = (ominv/gam).*rhs0 - 1/gam^2*ominv.*(Xt'*tmp);
                    
                else
                    % Initialize with all-zeros beta.
                    beta = zeros(p,1);
                end
%                 beta = zeros(p,1);
            else %  Warm-start with previous lambda.
                beta = B(:, j, ll-1);
            end
%             fprintf('norm b0: %1.3d \n', norm(beta))
%             fprintf('l0 b0: %d \n', nnz(beta))

            %+++++++++++++++++++++++++++++++++++++++++++++++++++++
            % Alternating direction method to update (theta, beta)
            %+++++++++++++++++++++++++++++++++++++++++++++++++++++

            for its = 1:maxits

                % Compute coefficient vector for elastic net step.
                d = 2*Xt'*(Yt*(theta/nt));

                % Update beta using proximal gradient step.
                b_old = beta;

                [beta, ~] = APG_ENbt(A, d, beta, lams(ll),L, eta, PGsteps, PGtol, true);
                

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
            end

            % Update Q and B.
            Q(:,j) = theta;
            B(:,j, ll) = beta;
        end

        %%
        %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        % Get classification statistics for (Q,B).
        %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        
        % Get prediction/scores.
        stats = predict(B(:,:,ll), [labs(vinds), Xv], C');
        mc(f, ll) = stats.mc;
        
        
        %%
        %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        % Validation scores.
        %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        %nnz(B)
        size(stats.l0)
        size(q)
        size(p)
        size(feat)
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

% average CV scores.
avg_score = mean(scores);

% choose lambda with best average score (break ties by taking largest ->
% most sparse discriminant vector).

% minidx = find(avg_score == minscore);
% lbest = max(minidx);
minscore = min(avg_score);
lbest = find(avg_score == minscore, 1, 'last');

% [~, lbest] = min(avg_score);


lambest = lams(lbest);


%% Solve with lambda = lam(lbest).

% Finished training lambda.
fprintf('Finished Training: lam = %d \n', lambest)

% Use full training set.
Xt = X(1:(n-pad), :);
Xt = normalize(Xt);
Yt = Y(1:(n-pad), :);

[B,Q] = SDAAPbt(Xt, Yt, Om, gam, lams(lbest),L, eta,  q, PGsteps, PGtol, maxits, tol, true);
