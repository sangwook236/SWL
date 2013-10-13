function prob = vm_pdf(dir_ang, mu, kappa)

% von Mises distribution (1-dimensional)
%
% dir_ang: a direction angle, [0 2*pi), [rad].
% mu: a mean direction angle, [rad].
% kappa: a concentration parameter, kappa >= 0.

prob = exp(kappa * cos(dir_ang - mu)) / (2 * pi * besseli(0, kappa));
