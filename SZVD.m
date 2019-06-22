function [DVs, its, pen_scal, N, w0] = SZVD(train, gamma, D, penalty, scaling, tol, maxits, beta, quiet)

% Applies SZVD heuristic for sparse zero-variance discriminant analysis to given training set. 
%==============================================================================================
% Input.
%==============================================================================================
%   train: training data set.
%   gamma: set of regularization parameters controlling l1-penalty.
%   D: dictionary/basis matrix.
%   penalty: controls whether to apply reweighting of l1-penalty (using sigma = within-class std devs)
%   scaling: whether to rescale data so each feature has variance 1.
%   tol: stopping tolerances used in the ADMM algorithm.
%   maxits: maximum number of iterations used in the ADMM algorithm.
%   beta: parameter for augmented Lagrangian term in the ADMM algorithm.
%==============================================================================================  
% Output.
%==============================================================================================
%   DVs: discriminant vectors.
%   its: % of iterations required to find DVs
%   pen_scal: weights used in reweighted l1-penalty.
%   N: basis for the null space of the sample within-class covariance.
%   w0: unpenalized zero-variance discriminants (initial solutions) plus B and W, etc.

%==============================================================================================
% Preprocess the training set.
%==============================================================================================
  
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Get covariance matrices and initial solutions.
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

% Get dimensions of the training set.
[~,p] = size(train);
p = p-1;
  
% Call ZVD to process training data.
get_DVs = 1;
w0 = ZVD(train, scaling, get_DVs);
  
% Normalize B (divide by the spectral norm)
[~, Btype] = size(w0.B);
if (Btype == 1)
    w0.B = w0.B/norm(w0.B);
else
    w0.B = 0.5*(w0.B+ w0.B');
    w0.B = w0.B/norm(w0.B);
end


% Extract scaling vector for weighted l1 penalty and diagonal penalty matrix.
if (penalty==1) % scaling vector is the standard deviations of each feature.
    s = sqrt(diag(w0.W));
elseif(penalty==0) % scaling vector is all-ones (i.e., no scaling)
    s = ones(p,1);
end

  
%==============================================================================================
% Initialization for the algorithm.
%==============================================================================================

% Initialize output.
DVs = zeros(p, w0.k-1);
its = zeros(w0.k-1, 1);

% Initalize objective matrix
if (Btype==1)
    B0 = w0.B;
else
    B0 = w0.N'*w0.B*w0.N;
    B0 = 0.5*(B0+ B0');
end

% Initialize N.
N = w0.N;

%==============================================================================================
% Find the DVs sequentially using ADMM. 
%==============================================================================================
  
  for i=1:(w0.k-1)
       % i, gamma(i)
      % Initial solutions.
      sols0.x = N'*D'*w0.dvs(:,i);
      sols0.y = w0.dvs(:,i);
      sols0.z = zeros(p,1);
      
      
      %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      %% Call ADMM solver.
      %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      [x,~,~, its] = SZVD_ADMM(B0,  N, D, sols0, s,  gamma(i), beta, tol, maxits, quiet);
      
      % Save output.
      DVs(:,i) = D*N*x;
      its(i) = its;
      
      if (quiet == 0)          
          fprintf('Found SZVD %g after %g its \n', i, its(i));
      end
    
    
        
      %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      %% Update N using QR.
      %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      if (i < (w0.k-1))
          % Project columns of N onto orthogonal complement of Nx.
          x = DVs(:,i);
          x = x/norm(x);
          
          Ntmp = N - x * (x'*N);
          
          % Call QR factorization to extract orthonormal basis for span(Ntmp)
          [Q,R] = qr(Ntmp);
          
          % Extract nonzero rows of R to get columns of Q to use as new N.
          R_rows = (abs(diag(R)) > 1e-6);
          
          % Use nontrivial columns of Q as updated N.
          N = Q(:, R_rows);
          
          % Update B0 according to the new basis N.
          B0 = N' * w0.B * N;
          B0 = 0.5*(B0+B0');
          
      end % End if.


   
  end % Close for loop.
  
  
%==============================================================================================
% Prep output.
%==============================================================================================
  
pen_scal=s;

end

