
function s = vec_shrink(v, a)
% Applies the soft thresholding shrinkage operator to v with tolerance a.
% That is, s is the vector with entries with absolute value v_i - a if
% |v_i| > a and zero otherwise, with sign pattern matching that of v.
%====================================================================
% Input.
%====================================================================
%   v: vector to be thresholded.
%   a: vector of tolerances.
%====================================================================
% Output.
%====================================================================
%	s: thresholded vector.


s = sign(v).*max(abs(v)-a, zeros(length(v), 1));

end