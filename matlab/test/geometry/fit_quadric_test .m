%addpath('../../src/geometry');

%-----------------------------------------------------------
% Elliptic paraboloid: x^2 / 3^2 + y^2 / 1^2 - z = 0.
%	z = x^2 / 9 + y^2.

[x, y] = meshgrid(linspace(-10, 10, 51), linspace(-10, 10, 51));
z = x.^2 / 9 + y.^2 + 2 .* randn(size(x));

coeffs = fit_quadric(x(:), y(:), z(:));
%coeffs / -coeffs(9)

Q11 = coeffs(1);
Q22 = coeffs(2);
Q33 = coeffs(3);
Q12 = coeffs(4);
Q23 = coeffs(5);
Q13 = coeffs(6);
Q14 = coeffs(7);
Q24 = coeffs(8);
Q34 = coeffs(9);
Q44 = coeffs(10);

% Approximation.
z_recon = -(Q11*x.^2 + Q22*y.^2 + Q33*z.^2 + Q12*x.*y + Q23*y.*z + Q13*z.*x + Q14*x + Q24*y + Q44) / Q34;

figure;
hold on;
mesh(x, y, z_recon);
plot3(x, y, z, '.');
hold off;

%-----------------------------------------------------------
% Ellipsoid: x^2 / 2^2 + y^2 / 3^2 + z^2 / 1^2 - 1 = 0.
%	z = +-sqrt(1 - x^2 / 4 - y^2 / 9).

%[x, y] = meshgrid(linspace(-2, 2, 51), linspace(-3, 3, 51));
[x, y] = meshgrid(linspace(-1, 1, 51), linspace(-2, 2, 51));
zp = sqrt(1 - x.^2 / 4 - y.^2 / 9) + 0.1 .* randn(size(x));
zn = -sqrt(1 - x.^2 / 4 - y.^2 / 9) + 0.1 .* randn(size(x));

coeffs = fit_quadric([x(:) ; x(:)], [y(:) ; y(:)], [zp(:) ; zn(:)]);
%coeffs / coeffs(3)

Q11 = coeffs(1);
Q22 = coeffs(2);
Q33 = coeffs(3);
Q12 = coeffs(4);
Q23 = coeffs(5);
Q13 = coeffs(6);
Q14 = coeffs(7);
Q24 = coeffs(8);
Q34 = coeffs(9);
Q44 = coeffs(10);

% Approximation.
zp_recon = sqrt(-(Q11*x.^2 + Q22*y.^2 + Q12*x.*y + Q23*y.*z + Q13*z.*x + Q14*x + Q24*y + Q34*z + Q44) / Q33);
zn_recon = -sqrt(-(Q11*x.^2 + Q22*y.^2 + Q12*x.*y + Q23*y.*z + Q13*z.*x + Q14*x + Q24*y + Q34*z + Q44) / Q33);

figure;
hold on;
mesh(x, y, zp_recon);
mesh(x, y, zn_recon);
plot3(x, y, zp, '.');
plot3(x, y, zn, '.');
hold off;
