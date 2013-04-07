function prob = vm2_pdf(phi, psi, mu, nu, kappa1, kappa2, kappa3)

% von Mises distribution on the torus (2-dimensional)
%
% phi: a angle on the torus, [0 2*pi], [rad].
% psi: a angle on the torus, [0 2*pi], [rad].
%   such an angle pair defines a point on the torus.
%   [ref] http://en.wikipedia.org/wiki/Bivariate_von_Mises_distribution
% mu: the mean for phi.
% nu: the mean for psi.
% kappa1: the concentration of phi.
% kappa2: the concentration of psi.
% kappa3: the correlation between kappa1 and kappa2.

prob = C(kappa1, kappa2, kappa3) * exp(kappa1 * cos(phi - mu) + kappa2 * cos(psi - nu) - kappa3 * cos(phi - mu - psi + nu)));

%--------------------------------------------------------------------
function normalization_constant = C(kappa1, kappa2, kappa3)

error('not yet implemented');

return;
