function coeffs = linear_regression(x, y)
% a * x + b * y + c = 0.
% coeffs = [a b c].

% NOTICE [caution] >> How to handle a case where a line is nearly vertical (b ~= 0, infinite slope).
%	- Can do something for exceptional cases in most cases.

X = [ones(size(x)) x];
coeffs = pinv(X) * y;

coeffs = [coeffs(2) -1 coeffs(1)];

return;

%-----------------------------------------------------------

X = [ones(size(x)) x];
coeffs = X \ y;

coeffs = [coeffs(2) -1 coeffs(1)];

return;

%-----------------------------------------------------------
% Use SVD => unstable.

% NOTICE [caution] >> Might compute incorrect results.
%	- Data normalization might be effective.

[U, S, V] = svd([x y ones(size(x))]);

coeffs = V(1,end)';

return;
