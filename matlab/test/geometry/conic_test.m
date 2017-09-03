%addpath('D:/work/SWL_github/matlab/src/geometry');

%-----------------------------------------------------------

% Conic (section): A*x^2 + B*x*y + C*y^2 + D*x + E*y + F = 0

%C_conic = [
%	A B/2 D/2
%	B/2 C E/2
%	D/2 E/2 F
%];
CC = [
	2*A B D
	B 2*C E
	D E 2*F
];
if rank(CC) < 3
	disp('[Warning] Degenerate conic.');
	return;
end;

discrimiant = B^2 - 4 * A * C;
if abs(discrimiant) < eps
	% Parabola.
elseif discrimiant < 0
	% Ellipse.
	% If A = C, circle.
else
	% Hyperbola.
	% If A + C = 0, rectanglar hyperbola.
end;
