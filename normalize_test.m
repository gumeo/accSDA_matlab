function x=normalize_test(x, mu, sig)
[n,~]=size(x);
x=x-ones(n,1)*mu;
%sig(~sig(:))=1;
x=x*diag(1./sig);
end