%addpath('D:/work/SWL_github/matlab/src/geometry');

use_draw_conic = true;

%-----------------------------------------------------------
% Easy example.

C1 = [
	1 0 0
	0 9 0
	0 0 -9
];

theta = pi / 4;
[C1_transformed, ABCDEF] = transform_conic(C1, [cos(theta) -sin(theta) ; sin(theta) cos(theta)], [0 ; 0]);

figure;
hold on;
if use_draw_conic
	draw_conic(ABCDEF, 'r-');
	draw_conic([1, 0, 1, 0, 0, -4], 'g-');
else
	A = ABCDEF(1);
	B = ABCDEF(2);
	C = ABCDEF(3);
	D = ABCDEF(4);
	E = ABCDEF(5);
	F = ABCDEF(6);

	ezplot(@(x,y) A * x^2 + B * x*y + C * y^2 + D * x + E * y + F);
	ezplot(@(x,y) x^2 + y^2 - 4);
end;
axis equal;
hold off;

%-----------------------------------------------------------

[a, b, theta, tx, ty] = deal(2, 1, pi / 4, 1, 0);
%[a, b, theta, tx, ty] = deal(2, 0.05, pi / 4, 1, 0);  % ezplot: bad drawing <- major axis / minor axis >> 1.

ABCDEF = ellipse2conic(a, b, theta, tx, ty);

[a0, b0, theta0, tx0, ty0] = conic2ellipse(ABCDEF);

figure;
hold on;
if use_draw_conic
	draw_conic(ABCDEF, 'r-');
	draw_conic([1, 0, 1, 0, 0, -2], 'g-');
else
	A = ABCDEF(1);
	B = ABCDEF(2);
	C = ABCDEF(3);
	D = ABCDEF(4);
	E = ABCDEF(5);
	F = ABCDEF(6);

	ezplot(@(x,y) A * x^2 + B * x*y + C * y^2 + D * x + E * y + F);
	ezplot(@(x,y) x^2 + y^2 - 2);
end;
axis equal;
hold off;

%-----------------------------------------------------------
% Intersection points between the ellipse with a line y = x.
%	A*x^2 + B*x*y + C*y^2 + D*x + E*y + F = 0.
%	y = x.

[a, b, theta, tx, ty] = deal(2, 1, pi / 4, 1, 0);
ABCDEF = ellipse2conic(a, b, theta, tx, ty);
A = ABCDEF(1);
B = ABCDEF(2);
C = ABCDEF(3);
D = ABCDEF(4);
E = ABCDEF(5);
F = ABCDEF(6);

a = A + B + C;
b = D + E;
c = F;
sqrt_discriminant = sqrt(b^2 - 4 * a * c);
x_sol1 = (-b + sqrt_discriminant) / (2 * a)
x_sol2 = (-b - sqrt_discriminant) / (2 * a)
