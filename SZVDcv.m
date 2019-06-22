function [DVs,  gambest] = SZVDcv(train, folds, gams,  beta,D, q, maxits, tol, ztol, feat, quiet)

% Applies proximal gradient algorithm with cross validation
% to the optimal scoring formulation of
% sparse discriminant analysis proposed by Clemmensen et al. 2011.
%
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Input
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% train.X: n by p data matrix.
% train.Y: n by K matrix of indicator variables (Yij = 1 if i in classs j)
% folds: number of folds to use in K-fold cross-validation.
% gams > 0: array of regularization parameters for l1 penalty.
% beta > 0: Augmented Lagrangian parameter.%
% D: penalty dictionary basis matrix
% q: desired number of discriminant vectors.
% maxits: number of iterations to run alternating direction alg.
% tol: stopping tolerance for alternating direction algorithm.
% ztol: rounding tolerance for truncating entries to 0.
% feat: maximum fraction of nonzero features desired in validation scheme.
% quiet: toggles display of iteration stats.
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
pad = 0; % Initialize number of padding observations.
if mod(n,folds) > 0
    % number of elements to duplicate.
    pad = ceil(n/folds)*folds - n;

    % duplicate elements of X and Y.
    X = [X; X(1:pad, :)];
    Y = [Y; Y(1:pad, :)];
end

% Recompute size.
[n, ~] = size(X);

% Randomly permute rows of X.
prm = randperm(n);
X = X(prm, :);
Y = Y(prm, :);

% Make Atrain.
Atrain = zeros(n, p+1);
for i = 1:n
    for j = 1:K
        if (Y(i,j) == 1)
            Atrain(i,1) = j;
        end
    end
end
Atrain(:, 2:(p+1)) = X;

% Sort lambdas in descending order (break ties by using largest lambda =
% sparsest vector).
%gams = sort(gams, 'descend');


%% Initialize cross-validation indices.

% Number of validation samples.
nv = n/folds;

% Initial validation indices.
vinds = (1:nv)';

% Initial training indices.
tinds = ((nv+1):n)';

% Get number of parameters to test.
ngam = length(gams);

% Validation scores.
scores = zeros(folds, ngam);

% Misclassification rate for each classifier.
mc = zeros(folds, ngam);


for f = 1 : folds

    %% Initialization.

    % Extract X and Y from train.
    Xt = X(tinds, :);
    %Yt = Y(tinds, :);
    At = Atrain(tinds, :);

    % Extract validation data.
    %Xv = X(vinds, :);
    %Yv = Y(vinds, :);
    Av = Atrain(vinds, :);


    % Get dimensions of training matrices.
    [nt, p] = size(Xt);

    % Call ZVD function to solve the unpenalized problem.
    get_DVs = 1;
    w0 = ZVD(At, 0, get_DVs);

    % Normalize B (divide by the spectral norm)
    if (size(w0.B, 2)==1) % B stored as vector.
        w0.B = w0.B/norm(w0.B);
    else % B stored as p by p symmetric matrix.
        w0.B = w0.B + w0.B'; % symmetrize.
        w0.B =w0.B/norm(w0.B); %normalize.
    end

    % Initalize objective matrix
    if (size(w0.B, 2)==1) % B is stored as a vector.
        B0 = w0.B;

    else % B is stored as a matrix.
        B0 = w0.N' * w0.B * w0.N;
        B0 = (B0+B0')/2;
    end

    % Initialize nullspace matrix.
    N0 = w0.N;

    % Number of gammas.
    num_gammas = length(gams);
    %DVs = zeros(p, K-1, num_gammas);

    %% Validation Loop.

    if (quiet == 0)
        fprintf('++++++++++++++++++++++++++++++++++++\n')
        fprintf('Fold %d \n', f)
        fprintf('++++++++++++++++++++++++++++++++++++\n')
    end

    %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    % Loop through potential regularization parameters.
    %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    for ll = 1:num_gammas

        %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        % Initialization.
        %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        % Initialize B and N.
        B = B0;
        N = N0;


        % Initialize B and Q.
        DVs = zeros(p, q);

        % Set x0 to be the unpenalized zero-variance discriminant vectors in Null(W0)
        if (size(B0,2) == 1) % B0 vector.

            % Compute (DN)'*(mu1-mu2)
            w = N0'*D'*B0;

            % Use normalized w as initial x.
            x0 = w/norm(w);

        else % B0 matrix,
            x0 = N0'*D'*w0.dvs(:,1);
        end

        % y is the unpenalized solution in the original space.
        % z is the all-zeros vector in the original space.

        %%
        %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        % Call Alternating Direction Method to solve SZVD problem.
        %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        for j = 1:q

            %+++++++++++++++++++++++++++++++++++++++++++++++++++++
            % Initialization.
            %+++++++++++++++++++++++++++++++++++++++++++++++++++++

            % Compute Qj (K by j, first j-1 scoring vectors, all-ones last col).
            %sols0.x = rand(length(x0),1);
            %sols0.y = rand(length(w0.dvs(:,j)),1);
            %sols0.z = zeros(p,1);

            sols0.x = x0;
            sols0.y = w0.dvs(:,j);
            sols0.z = zeros(p,1);
            quietADMM = 1;

            %+++++++++++++++++++++++++++++++++++++++++++++++++++++
            % Call ADMM solver.
            %+++++++++++++++++++++++++++++++++++++++++++++++++++++

            %% Call ADMM solver.
            s = ones(p,1);
            [tmpx, ~,~,~] = SZVD_ADMM(B,  N, D, sols0, s, gams(ll,j), beta, tol, maxits, quietADMM);

            % Extract j-th discriminant vector.
            DVs(:, j) = D*N*tmpx;
            %DVs(:, j,i) = tmpy;
            %DVs(:,j,i) = DVs(:,j,i)/norm(DVs(:,j,i));

            %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            % Update N and B for the newly found DV.
            %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            if (j < q)

                % Project columns of N onto orthogonal complement of Nx.

                x = DVs(:,j);
                x = x/norm(x);

                % Project N into orthogonal complement of span(x)
                Ntmp = N - x *(x'*N);

                % Call QR factorization to extract orthonormal basis for span(Ntmp)
                [Q,R] = qr(Ntmp);

                % Extract nonzero rows of R to get columns of Q to use as new N.
                R_rows = (abs(diag(R)) > 1e-6);

                % Use nontrivial columns of Q as updated N.
                N = Q(:, R_rows);

                % Update B0 according to the new basis N.
                B = N' * w0.B * N;
                B = 0.5*(B+B');


                % Update initial solutions in x direction by projecting next unpenalized ZVD vector.
                x0 = N'*(D'*w0.dvs(:,j+1));

            end % if j < q


        end % j = 1,2, ..., q.

        %%
        %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        % Get performance scores on the validation set.
        %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        % Call test_ZVD to get predictions, etc.
        %w0.mu

        %fprintf('getting predictions')
        [stats,~] = test_ZVD(DVs(:,:), Av, w0.means, w0.mu, 0);


        %%
        %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        % Validation scores.
        %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        % Round small entries to zero.
        DVs = DVs.*(ceil(abs(DVs) - ztol));

        if nnz(DVs) <= q*p*feat && nnz(DVs) >= 1 % if fraction nonzero features less than feat.
            % Use misclassification rate as validation score.
            scores(f, ll) = stats.mc;
        elseif nnz(DVs) < 0.5 % Trivial solution.
            scores(f, ll) = q*p; % Disqualify with largest possible value.
        else % Solution is not sparse enough, use most sparse as measure of quality instead.
            scores(f, ll) = nnz(DVs);
        end


        % Display iteration stats.
        if (quiet ==0)
            fprintf('f: %3d | ll: %3d | lam: %1.5e| ffeat: %1.5e | nfeat: %1.5e | mc: %1.5e | score: %1.5e \n', f,  ll, gams(ll), nnz(DVs)/(q*p), nnz(DVs), mc(f, ll), scores(f, ll))
        end


    end % For ll = 1:ngam.

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

% choose lambda with best average score.
[~, gbest] = min(avg_score);

gambest = gams(gbest,:);




%% Solve with lambda = lam(lbest).

% Finished training lambda.
%
% Loop until nontrivial solution is found.
trivsol = 1;
while trivsol == 1
    [DVs, ~, ~, ~, ~] = SZVD(Atrain, gambest, D, 0, 0, tol, maxits, beta, quietADMM);

     % Round small entries to zero.
     DVs = DVs.*(ceil(abs(DVs) - ztol));

     % Check for trivial solution.
     if nnz(DVs) == 0
         % If trivial solution, update gbest by one and update gambest.
         gbest = gbest + 1;
         gambest = gams(gbest,:);
     else
         % Have found a nontrivial solution.
         trivsol = 0;
     end
end


end
