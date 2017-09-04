function [a, b, theta, tx, ty] = conic2ellipse(ABCDEF)
% Input: A*x^2 + B*x*y + C*y^2 + D*x + E*y + F = 0.
% Output: An ellipse x^2 / a^2 + y^2 / b^2 = 1 rotated by rotation angle (theta) and translated by (tx, ty).
% REF [site] >> https://en.wikipedia.org/wiki/Ellipse

[a, b, theta, tx, ty] = deal([], [], [], [], []);

if rank(conic_poly2mat(ABCDEF)) < 3
	disp('[Warning] Degenerate conic.');
	return;
end;

A = ABCDEF(1);
B = ABCDEF(2);
C = ABCDEF(3);
D = ABCDEF(4);
E = ABCDEF(5);
F = ABCDEF(6);

discrimiant = B^2 - 4 * A * C;
if abs(discrimiant) < eps
	% Parabola.
elseif discrimiant < 0
	% Ellipse.

	tx = (2 * C * D - B * E) / discrimiant;
	ty = (2 * A * E - B * D) / discrimiant;

	% Assume a > b.
	KK = (A - C)^2 + B^2;
	sqrt_KK = sqrt(KK);
	PP = 2 * (A * E^2 + C * D^2 - B * D * E + discrimiant * F);
	a = -sqrt(PP * (A + C + sqrt_KK)) / discrimiant;
	b = -sqrt(PP * (A + C - sqrt_KK)) / discrimiant;

	if abs(B) < eps
		if A < C
			theta = 0;
		else
			theta = pi / 2;
		end;
	else
		%theta = atan2(C - A - sqrt_KK, B);  % [-pi, pi].
		theta = atan((C - A - sqrt_KK) / B);  % [-pi/2, pi/2].
	end;
else
	% Hyperbola.
	% If A + C = 0, rectanglar hyperbola.
end;
