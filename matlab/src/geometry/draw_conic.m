function [] = draw_conic(ABCDEF, plot_pattern)
% A*x^2 + B*x*y + C*y^2 + D*x + E*y + F = 0.
% REF [site] >> https://en.wikipedia.org/wiki/Conic_section

% NOTICE [caution] >> Sometime ezplot() shows a drawing of bad quality.
%	e.g.) When the ratio of an ellipse's major axis to its minor axis is very large. 

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
	ezplot(@(x,y) A * x.^2 + B * x*y + C * y.^2 + D*x + E*y + F, plot_pattern);
elseif discrimiant < 0
	% Ellipse.
	%ezplot(@(x,y) A * x.^2 + B * x*y + C * y.^2 + D*x + E*y + F, plot_pattern);
	draw_ellipse(ABCDEF, plot_pattern);
else
	% Hyperbola.
	% If A + C = 0, rectanglar hyperbola.
	ezplot(@(x,y) A * x.^2 + B * x*y + C * y.^2 + D*x + E*y + F, plot_pattern);
end;

return;

%-----------------------------------------------------------

function [] = draw_ellipse(ABCDEF, plot_pattern)
% A*x^2 + B*x*y + C*y^2 + D*x + E*y + F = 0.

[a, b, theta, tx, ty] = conic2ellipse(ABCDEF);

cos_th = cos(theta);
sin_th = sin(theta);
T = [cos_th -sin_th tx ; sin_th cos_th ty ; 0 0 1];

angle = linspace(0, 2*pi);
ellipse = T * [a * cos(angle) ; b * sin(angle) ; ones(size(angle))];

% Draw.
plot(ellipse(1,:), ellipse(2,:), plot_pattern);

return;
