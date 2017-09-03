function ABCDEF = fit_conic(x, y)
% Conic section: A*x^2 + B*x*y + C*y^2 + D*x + E*y + F = 0.
% More than 5 points are required.

A = [x.^2 x.*y y.^2 x y ones(size(x))];
[U, S, V] = svd(A);
ABCDEF = V(:,end)';

%if abs(ABCDEF(1)) > eps
%	ABCDEF = ABCDEF / ABCDEF(1);
%elseif abs(ABCDEF(3)) > eps
%	ABCDEF = ABCDEF / ABCDEF(3);
%end;
