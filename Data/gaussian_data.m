function [obs,mu,Sigma] = gaussian_data(p,r,k,N)
% GAUSSIAN_DATA generates synthetic data for benchmarking LDA heuristics.
% Samples Gaussian observations from the distributions mvrnd(mu, Sigma)where
% mu has entries in the i_th block of 100 consecutive entries equal to .7 &
% Sigma has all off-diagonal entries equal to r. 
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%Input. 
% p: dimension of observations to generate.
% k: number of classes to generate.
% r: desired value of constant covariance between features.
% N: vector of class sizes.
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%Output. obs = obs, mu = mu, Sigma = Sigma
  %   obs: observation matrix with rows sampled according to N(mu, Sigma)
  %   mu: matrix of mean vectors (i-th column is mean of i-th class).
  %   Sigma: covariance matrix with all off-diagonal entries equal to r
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

%set size of diagonal blocks of Sigma.
l = 100;

%initialize mean vector mu and covariance matrix Sigma
mu = zeros(p, k); 
Sigma = zeros(p,p);

%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%Form the matrix of mean vectors mu. 
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

%Each class mean satisfies mu_{jk} = 0.7 if j in [(k-1)*100+1, k*100]
for i = 1:k
    mu(((i-1)*l +1):(l*i), i)= 0.7*ones(l,1);
end

%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%Form the covariance matrix Sigma
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

%Sigma has all off-diagonal entries equal to r.
%Set all entries equal to r
for i = 1:p
    for j=1:p
        if j==i
            Sigma(i,j)=1;
        else
            Sigma(i,j)=r;
        end
    end
end

%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%Generate sum(N) observation using mu and sigma
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

% Initialize observations matrix. 
%obs = zeros(sum(N), p+1);

%Initialize start/end of current class.
Start = 1;
End   = N(1);

obs=zeros(sum(N),p+1);

% Generate i-th class
for i = 1:k

    %sample observations. 
    obs(Start:End,1) = i; %
    
%     disp(size(obs))
%     disp(Start)
%     disp(End)
   %   disp(p)
   %   disp(size(Sigma))
   %   disp(size(mu))
   %   disp(p)
    obs(Start:End,2:(p+1)) = mvnrnd( transpose(mu(:,i)),Sigma,N(i) );%<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    % Update block positions.
     if i<k
         Start = End + 1;
         End = End + N(i+1);
     end
end

    








