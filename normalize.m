function [x, mu, sig] = normalize(x)
[n,~]=size(x);
mu = mean(x);
x=x-ones(n,1)*mu;
sig=std(x);
%sig(~sig(:))=1;
x=x*diag(1./sig);
end
