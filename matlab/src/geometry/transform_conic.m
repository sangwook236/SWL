function [C_transformed, ABCDEF] = transform_conic(C_conic, R, t)
% Conic section.
%	x^T * C_conic * x = 0 <= A*x^2 + B*x*y + C*y^2 + D*x + E*y + F = 0.
%	C_conic = [
%		A B/2 D/2
%		B/2 C E/2
%		D/2 E/2 F
%	]
% R: 2D rotation transformation: R = R(theta) = [cos(theta) -sin(theta) ; sin(theta) cos(theta)].
% t: 2D translation vector: t = [tx ; ty].
%	2D homogeneous transformation, T = [ R t ; 0 1 ].
%	T^-1 = [ R^T -R^T*t ; 0 1 ].
% REF [site] >> https://en.wikipedia.org/wiki/Conic_section

if nargin < 2
	R = eye(3);
end;
if nargin < 3
	t = [0 ; 0];
end;

% x' = T * x.
% x^T * C_conic * x = (T^-1 * x')^T * C_conic * (T^-1 * x') = x'^T * T^-T * C_conic * T^-1 * x' = x'^T * C_conic' * x' = 0.
% C_conic' = T^-T * C_conic * T^-1.

% T = [cos_th -sin_th tx ; sin_th cos_th ty ; 0 0 1];
% invT = [cos_th sin_th -tx*cos_th-ty*sin_th ; -sin_th cos_th tx*sin_th-ty*cos_th ; 0 0 1];

%invT = inv([R t ; 0 0 1]);
invT = [R' -R'*t ; 0 0 1];
C_transformed = invT' * C_conic * invT;
C_transformed = (C_transformed + C_transformed') * 0.5;

if nargout > 1
	ABCDEF = conic_mat2poly(C_transformed);
	%[A, B, C, D, E, F] = conic_mat2poly(C_transformed);
end;
