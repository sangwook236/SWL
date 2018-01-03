function coeffs = fit_quadric(x, y, z, method)
% Output: coeffs = [ Q11 Q22 Q33 Q12 Q23 Q13 Q14 Q24 Q34 Q44 ].
%	Q11*x^2 + Q22*y^2 + Q33*z^2 + Q12*x*y + Q23*y*z + Q13*z*x + Q14*x + Q24*y + Q34*z + Q44 = 0.
% More than 9 points are required.
% REF [site] >> https://en.wikipedia.org/wiki/Quadric

if nargin < 4 | strcmpi(method, 'sampson')
	coeffs = fit_quadric_by_sampson_approximation(x, y, z);
elseif strcmpi(method, 'regression')
	coeffs = fit_quadric_by_linear_regression(x, y, z);
else
	error('Invalid fitting method.');
end;

return;

%-----------------------------------------------------------

function coeffs = fit_quadric_by_linear_regression(x, y, z)
% Use linear regression.
% REF [book] >> "Multiple View Geometry in Computer Vision", p. 31.

[U, S, V] = svd([x.^2 y.^2 z.^2 x.*y y.*z z.*x x y z ones(size(x))]);
coeffs = V(:,end)';

%if abs(coeffs(1)) > eps
%	coeffs = coeffs / coeffs(1);
%elseif abs(coeffs(2)) > eps
%	coeffs = coeffs / coeffs(2);
%elseif abs(coeffs(3)) > eps
%	coeffs = coeffs / coeffs(3);
%end;

return;

%-----------------------------------------------------------

function coeffs = fit_quadric_by_sampson_approximation(x, y, z)
% Use the Sampson approximation to the geometric distance for a quadric.
% REF [book] >> "Multiple View Geometry in Computer Vision", p. 99.

coeffs_init = fit_quadric_by_linear_regression(x, y, z);

options = optimset('TolX', 1e-6, 'TolFun', 1e-6);
coeffs = fminsearch(@quadric_sampson_approximation_cost, coeffs_init, options, [x y z ones(size(x))]');

return;

%-----------------------------------------------------------

function cost = quadric_sampson_approximation_cost(coeffs, X)

Q = quadric_poly2mat(coeffs);
len = size(X, 2);
cost = 0;
for ii = 1:len
	QX = Q * X(:,ii);
	cost = cost + 0.25 * (X(:,ii)' * QX)^2 / (QX(1)^2 + QX(2)^2 + QX(3)^2);
end;

return;
