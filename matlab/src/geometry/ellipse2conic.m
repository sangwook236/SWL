function ABCDEF = ellipse2conic(a, b, theta, tx, ty)
% Input: An ellipse x^2 / a^2 + y^2 / b^2 = 1 rotated by rotation angle (theta) and translated by (tx, ty).
% Output: A*x^2 + B*x*y + C*y^2 + D*x + E*y + F = 0.
% REF [site] >> https://en.wikipedia.org/wiki/Ellipse

a2 = a^2;
b2 = b^2;
sin_th = sin(theta);
cos_th = cos(theta);

C_conic = [
	b2 0 0
	0 a2 0
	0 0 -a2*b2
];

[DD, ABCDEF] = transform_conic(C_conic, [cos_th -sin_th ; sin_th cos_th], [tx ; ty]);
