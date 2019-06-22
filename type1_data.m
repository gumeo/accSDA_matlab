%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%Generate synthetic data sets of the form used in the first set of 
%experiments.

%Samples Gaussian observations from the distributions mvrnd(mu, Sigma)where
%mu has entries in the i_th block of 100 consecutive entries equal to .7 &
%Sigma has all off-diagonal entries equal to r. 

% function main
% 
% end


%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%Constructing type1_data(p,r,k,N)
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%Input. 
    %p: dimension of observations to generate.
    %k: number of classes to generate.
    %r: desired value of constant covariance between features.
    %N: vector of class sizes.
%Output. obs = obs, mu = mu, Sigma = Sigma
  %   obs: observation matrix with rows sampled according to N(mu, Sigma)
  %   mu: matrix of mean vectors (i-th column is mean of i-th class).
  %   Sigma: covariance matrix with all off-diagonal entries equal to r
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

function [obs,mu,Sigma]=type1_data(p,r,k,N)
%%%%>>>>> p>l*k??? <<<<<

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

    
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%Output generated data, mean vector, mean vectors, and covariance matrix.
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

%=return(list(obs=obs, mu=mu, Sigma=Sigma))
end
%% Function type 2_data(p,k,r,N)
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%Generates synthetic data sets of the form used in the second set of
%experiments

%Samples Gaussian observations from the distributions N(mu_i, Sigma) where 
%mu_i has entries in i-th block of 100 consecutive entries equal to 0.7 and
%Sigma is block matrix with entries in each 100x100 diagonal block taking 
%value  r^|i-j| in the (i,j)th entry
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

function[obs, mu, Sigma] = type2_data(p,k,r,N)


    %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    %Input
    %   p: dimension of observations to generate.
    %   k: number of classes to generate.
    %   r: constant for computing covariance between any two features. 
    %   N: vector of class sizes. 
    
    %Output. obs = obs, mu=mu, Sigma=Sigma
    %   obs: observation matrix with rows sampled according to N(mu,Sigma)
    %   mu: matrix of mean vectors (i-th column is mean of i-th class).
    %   Sigma: desired block covariance matrix. 
    %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    %Initialize mean vector mu and covariance matrix Sigma.
    mu = zeros(p, k);
    Sigma = zeros(p, p);
    
    %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    % Form mu
    % Each class mean satisfies mu_{jk} = 0.7 if j in [(k-1)*100+1,k*100]
    %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    for i = 1:k
        mu(((i-1)*100+1):(100*i),i) = 0.7*ones(100,1);
    end
    
    %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    %Form Sigma
    %Sigma is block diagonal with 100*100 diagonal blocks with (i,j)th
    %value 
    %r^|i-j|
    %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    %Get number of blocks (requires p to be multiple of 100)
    Nb = p./100;
   
    for b = 1:Nb 
        %Compute the b-th diagonal block.
        %compute the nonzeore block entries.
        for i = 1:100
            for j = 1:100 
                Sigma((b-1)*100+i, (b-1)*100 +j) = r^(abs(i-j));
                
                %Fix diagonal entries to be 1 regardless of value of r. 
                if i==j
                    Sigma((b-1)*100+i, (b-1)*100+j) = 1;
                end
            end
        end
    end
    
    %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    % Generate sum(N) observations using mu and sigma.
    %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    % Initialize observations matrix. 
    obs = zeros(sum(N),p+1);
    
    %Initialize Start/End of current class.
    Start = 1;
    End = N(1);
    
    %Generate ith class 
    for i = 1:k 
        %Sample observations
        obs(Start:End, 1) = i;
        obs(Start:End, 2:(p+1)) = mvnrnd(transpos(mu(:,i)), Sigma, N(i));
        
        %Updat block positions. 
        if i < k
            Start = End + 1;
            End = End + N(i+1);
        end
    end
    
    %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    %Output generated data, mean vectors, and covariance matrix. 
    %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    %return(list(obs = obs, mu=mu, Sigma=Sigma))

end

 
%% Function typeT_data(p,r,k,N)
% 
% %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%     
%     %Input
%     %   p: dimension of observations to generate.
%     %   k: number of classes to generate.
%     %   r: desired value of constant covariance between features. 
%     %   N: vector of class sizes. 
%     
%     %Output. obs = obs, mu=mu, Sigma=Sigma
%     %   obs: observation matrix with rows sampled according to N(mu,Sigma)
%     %   mu: matrix of mean vectors (i-th column is mean of ith class)
%     %   Sigma: covariance matrix with all off-diagonal entries equal to r. 
% %+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% function[obs, mu, Sigma] = typeT_data(p,k,r,N)
% % Set size of diagonal block of Sigma.
% l=100; 
% 
% % Initialize mean vectors mu and covariance matrix Sigma. 
% nrow = p;
% ncol =k; 
% mu = zeros(nrow, ncol);
% ncol = p;
% Sigma = zeros(nrow, ncol);
% 
% %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% % Form mu.
% % Each class mean satisfies mu_{jk} = 0.7 if j in [(k-1)*100+1,k*100]
% %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% 
% for i = 1:k
%     mu(((i-1)*100+1):(100*i),i) = 0.7*ones(1,l);
% end
% 
% %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% %Form the covariance matrix Sigma
% %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% 
% % Sigma has all off-diagonal entries equal to r.
% % set all entries equal to r
% 
% for i = 1: nrow
%         for j= 1:ncol
%             if j == i 
%                 Sigma(i,j) = 1;
%             else
%                 Sigma(i,j) = r;
%             end
%         end
% end
% 
% %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% % Generate sum(N) observations using mu and Sigma.
% %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% 
% % Initialize observations matrix. 
% 
% obs = zero(sum(N),p+1);
% 
% %Initialize Start/End of current class.
% 
% Start = 1;
% End = N(1);
% 
% % Generate i-th class
% for i = 1:k
%     %sample observations
%     obs(Start:End,1) = i;
%     %make class 1.
%     obs(start:end, 2:(p+1)) =%tdistgen(n=N(i),delta=mu(:i),sigma=Sigma,df=1);
%     
%     %Update block positions
%     %if i<k 
%         %Start= End+1;
%         %End = End + N(i+1);
%     %end
% %end    
%     
% 
% 
% %++++++++++++++++++++++++++++++++++++++++
% % Output generated data, mean vector, and covariance matrix

% %end







