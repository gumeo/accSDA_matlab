function w0 = ZVD(A, scaling, get_DVs)
% Applies ZVD heuristic to given training set encoded in A.
%====================================================================
% Input.
%====================================================================
% A: matrix containing given data set (and labels)
% scaling: whether to rescale data so each feature has variance 1.
% get_DVs: whether to obtain unpenalized zero-variance discriminant vectors.
%====================================================================
% Output.
%==================================================================== 
% w0 containing
%		dvs: discriminant vectors (optional).
%		B: sample between-class covariance.
%		W: sample within-class covariance.
%		N: basis for the null space of the sample within-class covariance.  %
%		mu: training mean and variance scaling/centring terms
%		means: vectors of sample class-means.
%       k: number of classes in given data set.
%		labels: list of classes.
%		obs: matrix of data observations.
%		class_obs: matrices of observations of each class.
%====================================================================


%====================================================================
% Process training data.
% Extract labels, center and normalize.
%====================================================================

% Extract class labels.
classes = A(:,1);

% Get input dimensions.
[n,p] = size(A);
p = p-1;

% Extract observations.
X = A(:, 2:(p+1));

% Center the data.
mu = mean(X)';
%muz = size(mu);
X = X - ones(n,1)* mu';

%% Scale the data so each feature has variance equal to 1.
if (scaling == 1)
    % Compute standard deviation of each feature.
    sig = std(X);
    
    % Divide each feature by its standard deviation to ensure each feature has variance equal to one.
    X = X * diag(1./sig);
end

%====================================================================
% Initialize scatter matrices, etc.
%====================================================================

% Get number of classes.
K = max(classes);


% Initialize matrix of within-class means.
classMeans = zeros(p, K);

% Initialize within-class covariance.
W = zeros(p);

% Initialize between-class covariance (if more than 2 classes)
if (K>2)
    B = zeros(p);   
end

class_sizes = zeros(K,1);

%====================================================================
% Compute within-class scatter matrices.
%====================================================================

for i = 1: K
    
    % Find indices of observations in the current class.
    class_inds = (classes == i);
    
    % Make new object in class_obs corresponding to the observations in this class.
    class_obs = X(class_inds, :);
    
    % Get number of observations in this class.
    ni = size(class_obs, 1);
    class_sizes(i) = ni;
    
    % Compute within-class means.
    classMeans(:,i) = mean(class_obs);
    
    % Compute sample class-scatter matrix.    
    Xc = class_obs - ones(ni,1)*classMeans(:,i)';
          
    % Update W.
    W = W + Xc'*Xc;                

end % End for (k).

% Symmetrize W.
W = (W + W')/2;

%====================================================================
% Compute B 
%====================================================================

if (K==2) % (K=2 case) (Store as a p by 1 matrix/vector)
    B = classMeans(:,1)-classMeans(:,2);    

else % K > 2 case.
    
    % Make partition matrix.    
    Y = zeros(n,K);
    for i = 1:n
        Y(i, classes(i)) = 1;
    end
    
    % Set scatter matrix B.
    XY = X'*Y;
    B = XY*((Y'*Y)\XY');    
    
    % Symmetrize B.
    B = (B + B')/2;

end


%====================================================================
% Find the null-vectors of W.
%====================================================================
%fprintf('find null basis')
N = null(W);
%fprintf('found null basis')

%====================================================================
% Find ZVDs (if GetDVs = true)
%====================================================================
if (get_DVs==1)
    if (K==2)
        % Compute max eigenvector of N'*B*N
        w = N*(N'*B);
        w = w/norm(w);
    else
        % Compute K-1 nontrivial eigenvectors of N'*B*N.
        [w,~] = eigs(N'*B*N, K-1, 'LM');
        
        % Project back to the original space.
        w = N * w;
    end
        
        
end
        
%====================================================================
% Prep output.
%====================================================================

% Output scaling/centring terms (if used).
if (scaling==1)    
    mus.mu = mu;
    mus.sig= sig;
end

% Output discriminant vectors (if found).
if (get_DVs ==1)
    w0.dvs = w;
end

% Output everything else.
w0.B = B;
w0.W = W;
w0.N = N;
if scaling == 1
    w0.mu =mus;
else
    w0.mu = mu;
end
w0.means = classMeans;
w0.k= K;
w0.labels=classes;
w0.obs = X;
w0.ni = class_sizes;

