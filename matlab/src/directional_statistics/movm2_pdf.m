function prob = movm2_pdf(phi, psi, mu, nu, kappa1, kappa2, kappa3, alpha)

% a mixture of von Mises distributions on the torus (2-dimensional)
%
% phi: a angle on the torus, [0 2*pi], [rad].
% psi: a angle on the torus, [0 2*pi], [rad].
%   such an angle pair defines a point on the torus.
%   [ref] http://en.wikipedia.org/wiki/Bivariate_von_Mises_distribution
% mu: the means for phi.
% nu: the means for psi.
% kappa1: the concentrations of phi.
% kappa2: the concentrations of psi.
% kappa3: the correlations between kappa1 and kappa2.
% alpha: mixing coefficents, sum(alpha) = 1.

num1 = length(mu);
%num2 = length(nu);
%num3 = length(kappa1);
%num4 = length(kappa2);
%num5 = length(kappa3);
%num6 = length(alpha);

%if num1 ~= num2 || num1 ~= num3 || num1 ~= num4 || num1 ~= num5 || num1 ~= num6
%	error('the number of mixture components is un-matched ...');
%end;

prob = 0;
for ii = 1:num1
	prob = prob + alpha(ii) * vm2_pdf(phi, psi, mu(ii), nu(ii), kappa1(ii), kappa2(ii), kappa3(ii));
end;
