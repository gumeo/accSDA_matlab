function [classMeans, K, M, R]=calcClassMeans(train)
% CALCCLASSMEANS Calculates class-means and covariance matrices.
% calculates class-means and factorization of between/within-class 
% covariance matrices for use in penzda.
% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% INPUT.
% train - training data set. First column contains labels.
% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% OUTPUT.
% classMeans - matrix of class-means.
% K - number of classes in training data.
% M - factor of within-class covariance matrix W = M'*M.
% R - factor of between-class covariance matrix B = R'*R.
% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

% Extract labels.
classes=train(:,1);
[n,p]=size(train);

%Extract observations 
X=train(:,2:p);

labels=unique(classes); 
K=length(labels);

%Initiate matrix of class means.
p=p-1;
classMeans=zeros(p,K);

% Initialize matrix factorization between-class covariance.s
R=zeros(K,p);

%for each class, make an object in the list containing only the obs of that
%class and update the between and within-class sample
M=zeros(n,p);

for i=1:K    
    
    % Extract observations from class i.
    class_obs=X(classes==labels(i),:);
    
    %Get the number of obs in that class (the number of rows)
    ni=size(class_obs,1);
    
    %Compute within-class mean
    classMeans(:,i)=mean(class_obs);

    %Update W and R.
    M(classes == labels(i),:) =class_obs-ones(ni,1)*classMeans(:,i)';
    R(i,:)= sqrt(ni)*classMeans(:,i)';
end

