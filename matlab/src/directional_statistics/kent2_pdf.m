function prob = kent2_pdf(theta, phi, kappa, beta, gamma1, gamma2, gamma3)

% Kent distribution (2-dimensional)
%
% theta: an inclination(polar) angle, [0 pi], [rad].
% phi: an azimuthal angle, [0 2*pi), [rad].
%   [ref] http://en.wikipedia.org/wiki/Spherical_coordinate_system
% kappa: a concentration parameter, kappa >= 0.
%	The concentration of the density increases with kappa.
% beta: ellipticity of the equal probability contours of the distribution.
%	The ellipticity increases with beta.
%	If beta = 0, the Kent distribution becomes the von Mises-Fisher distribution on the 2D sphere.
% gamma1: the mean direction.
% gamma2: the main axis of the elliptical equal probability contours.
% gamma3: the secondary axis of the elliptical equal probability contours.
%	The 3x3 matrix, [ gamma1 gamma2 gamma3 ] must be orthogonal.
%
% In the 2-dimensional case:
% The Kent distribution, also known as the 5-parameter Fisher-Bingham distribution, is a distribution on the 2D sphere (the surface of the 3D ball).
% It is the 2D member of a larger class of N-dimensional distributions called the Fisher-Bingham distributions.

%dim = length(x);
%dim1 = length(gamma1);
%dim2 = length(gamma2);
%dim3 = length(gamma3);

%if dim ~= dim1 || dim ~= dim2 || dim ~= dim3
%	error('dimensions are un-matched ...');
%end;

%if dim ~= rank([gamma1 gamma2 gamma3])
%	error('orthogonality of gamma1, gamma2, & gamma3 is un-satisfied ...');
%end;

% x: a unit direction vector on 2-dimensional sphere, norm(x) = 1, column-major vector.
x = [ sin(theta)*cos(phi) ; sin(theta)*sin(phi) ; cos(theta) ];

prob = exp(kappa * dot(x, gamma1) + beta * (dot(x, gamma2)^2 - dot(x, gamma3)^2)) * sqrt((kappa - 2 * beta) * (kappa + 2 * beta)) / ( 2 * pi * exp(kappa));
