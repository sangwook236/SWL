function prob = vmf2_pdf(theta, phi, mu, kappa)

% von Mises-Fisher distributions (2-dimensional)
%
% theta: an inclination(polar) angle, [0 pi], [rad].
% phi: an azimuthal angle, [0 2*pi), [rad].
%   [ref] http://en.wikipedia.org/wiki/Spherical_coordinate_system
% mu: a mean direction vector, norm(mu) = 1, column-major vector.
% kappa: a concentration parameter, kappa >= 0.

%dim = 2;
%dim1 = length(mu);

%if dim ~= dim1
%	error('dimensions are un-matched ...');
%end;

% x: a unit direction vector on 2-dimensional sphere, norm(x) = 1, column-major vector.
x = [ sin(theta)*cos(phi) ; sin(theta)*sin(phi) ; cos(theta) ];
%mu = [ sin(mu_theta)*cos(mu_phi) ; sin(mu_theta)*sin(mu_phi) ; cos(mu_theta) ];

prob = (kappa / 2)^(dim/2 - 1) * exp(kappa * dot(x, mu)) / (gamma(dim / 2) * besseli(dim/2 - 1, kappa));
