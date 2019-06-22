function [val_w, DVs, gamma, gammas, max_gamma, its, w0, x0] = SZVD_Val(Atrain, Aval,  D,  k, num_gammas, gmults, sparsity_pen, scaling, penalty, beta, tol, maxits, quiet)
%% Performs SZVD with validation to choose regularization parameter gamma.
%====================================================================
% Input.
%====================================================================
%   Atrain: training data matrix
%   Aval: validation matrix.
%   k: number of classes within training and validation sets.
%   num_gammas: number of gammas to train on.
%   g_mults = (c_min, c_max): parameters defining range of gammas to train g_max*(c_min, c_max)
%   D: dictionary/basis matrix.
%   sparsity_pen: weight defining validation criteria as weighted sum of misclassification error and
%       cardinality of discriminant vectors.
%   scaling: whether to rescale data so each feature has variance 1.
%   penalty: controls whether to apply reweighting of l1-penalty (using sigma = within-class std devs)
%   beta: parameter for augmented Lagrangian term in the ADMM algorithm.
%   tol: stopping tolerances used in the ADMM algorithm.
%   maxits: maximum number of iterations used in the ADMM algorithm.
%   quiet: controls display of intermediate results.
%====================================================================
% Output.
%====================================================================
%   DVs: discriminant  vectors for best choice of gamma.
%   gamma: choice of gamma minimizing validation criterion.
%	gammas: set of all gammas considered.
% 	max_gamma: upper bound on gamma used.
%	its: number of iterations used for each gamma.
%   w0: unpenalized zero-variance discriminants (initial solutions) plus B and W, etc. from ZVD.
%	x0: initial solutions used.

%====================================================================
% Initialization.
%====================================================================
% Get dimensions of the training set.
%n = size(Atrain,1);
p = size(Atrain,2) - 1;

%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Compute penalty term for estimating range of regularization
% parameter values.
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

% Call ZVD function to solve the unpenalized problem.
get_DVs = 1;
w0 = ZVD(Atrain, scaling, get_DVs);

% Extract scaling vector for weighted l1 penalty and diagonal penalty matrix.
if (penalty==1) % scaling vector is the std deviations of each feature.
    s = sqrt(diag(w0.W));
elseif(penalty==0) % scaling vector is all-ones (i.e., no scaling)
    s = ones(p,1);
end

% Save scaling vector to w0.
w0.s = s;


%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%% Compute range of sensible parameter values.
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

%% Normalize B (divide by the spectral norm)
if (size(w0.B, 2)==1) % B stored as vector.
    w0.B = w0.B/norm(w0.B);
else % B stored as p by p symmetric matrix.
    w0.B = w0.B + w0.B'; % symmetrize.
    w0.B =w0.B/norm(w0.B); %normalize.
    
end

%% Compute ratio of max gen eigenvalue and l1 norm of the first ZVD to get "bound" on gamma.
if (size(w0.B, 2)==1)% B stored as vector.
    max_gamma =  ((w0.dvs)'*w0.B)^2/sum(abs(s.*(D * w0.dvs)));
    
else % B stored as matrix.
    
    % Allocate gammas.
    max_gamma = zeros(k-1, 1);
    
    % Define max gamma to be value forcing the objective equal to zero
    % for each zero-variance DV.
    for i = 1:(k-1)
        max_gamma(i) = w0.dvs(:,i)'*w0.B*w0.dvs(:,i)/norm(s.*(D*w0.dvs(:,i)), 1);
    end
    
end

% Generate range of gammas to choose from.
gammas = zeros(num_gammas, k-1);
for i = 1:(k-1)
    gammas(:,i) = linspace(gmults(1)*max_gamma(i), gmults(2)*max_gamma(i), num_gammas)';
end


%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Initialize the validation scores.
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
val_scores = zeros(num_gammas, 1);
%   mc_ind = 1  ;
%   l0_ind = 1;
best_ind = 1;
%   min_mc = 1;
%   min_l0 = p+1;

triv=0;

%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Initial matrices.
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

% Initalize objective matrix
if (size(w0.B, 2)==1) % B is stored as a vector.
    B0 = w0.B;
    
else % B is stored as a matrix.
    B0 = w0.N' * w0.B * w0.N;
    B0 = (B0+B0')/2;
end

% Initialize nullspace matrix.
N0 = w0.N;

% Initialize DVs and iteration lists.
DVs = zeros(p, k-1, num_gammas);
its = zeros(k-1, num_gammas);



%====================================================================
% For each gamma, calculate ZVDs and corresponding validation scores.
%====================================================================
for i =1:num_gammas
    
    %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    % Initialization.
    %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    % Initialize B and N.
    B = B0;
    N = N0;
    
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
    
    %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    %%% Get DVs
    %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    for j = 1:(k-1)
        
        % Initial solutions.
        sols0.x = x0;
        sols0.y = w0.dvs(:,j);
        sols0.z = zeros(p,1);        
        
        quietADMM = 1;
        %% Call ADMM solver.
        [tmpx, ~,~, tmpits] = SZVD_ADMM(B,  N, D, sols0, s, gammas(i,j), beta, tol, maxits, quietADMM);
        
        % Extract j-th discriminant vector.
        DVs(:, j,i) = D*N*tmpx;
        %DVs(:, j,i) = tmpy;
        %DVs(:,j,i) = DVs(:,j,i)/norm(DVs(:,j,i));
        
        % Record number of iterations to convergence.
        its(j,i) = tmpits;
        
        %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        % Update N and B for the newly found DV.
        %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        if (j < (w0.k-1))
            
            % Project columns of N onto orthogonal complement of Nx.
            x = DVs(:,j, i);
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
            
        end % if j < k-1
        
    end % for j = 1:k-1.
    
    %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    % Get performance scores on the validation set.
    %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    % Call test_ZVD to get predictions, etc.
    %w0.mu
    [stats,~] = test_ZVD(DVs(:,:,i), Aval, w0.means, w0.mu, scaling);
    
    %% Update the cross-validation score for this choice of gamma.
    
    % If gamma induces the trivial solution, disqualify gamma by assigning
    % large enough penalty that it can't possibly be chosen.
    if (sum(stats.l0) < 3)
        val_scores(i) = 100*size(Aval,1);
        triv=1;
        
    else
        % if all discriminant vectors are nontrivial use the formula:
        % e_q = %misclassified + % nonzero.
        val_scores(i) = stats.mc + sparsity_pen*sum(stats.l0)/(p*(k-1));
    end
    
    
    %% Update the best gamma so far.
    % Compare to existing proposed gammas and save best so far.
    if (val_scores(i) <= val_scores(best_ind))
        best_ind = i;
    end
    
    %     % Record sparsest nontrivial solution so far.
    %     if (min(SZVD_res$stats$l0) > 3 && SZVD_res$stats$l0 < min_l0){
    %       l0_ind = i
    %       l0_x = DVs[[i]]
    %       min_l0 = SZVD_res$stats$l0
    %     end
    %
    %     % Record best (in terms of misclassification error) so far.
    %     if (SZVD_res$stats$mc <= min_mc)
    %       mc_ind = i
    %       mc_x = DVs[[i]]
    %       min_mc = SZVD_res$stats$mc
    %     end
    
    
    % Display current iteration stats.
    if (quiet==0)
        fprintf('it = %g, val_score= %g, mc=%g, l0=%g, its=%g \n', i, val_scores(i), stats.mc, sum(stats.l0), mean(its(:,i)));
    end
    
    % Terminate if a trivial solution has been found.
    if (triv==1)
        break
    end
    
end  % For i = 1:gammas.


% Export discriminant vectors found using validation.
val_w = DVs(:,:, best_ind);
gamma = gammas(best_ind,:);


