function prob = movmf2_pdf(theta, phi, mu, kappa, alpha)

% a mixture of von Mises-Fisher distributions (2-dimensional)
%
% theta: an inclination(polar) angle, [0 pi], [rad].
% phi: an azimuthal angle, [0 2*pi), [rad].
%   [ref] http://en.wikipedia.org/wiki/Spherical_coordinate_system
% mu: mean direction vectors, norm(mu(:,i)) = 1, column-major vector.
% kappa: concentration parameters, kappa(i) >= 0.
% alpha: mixing coefficents, sum(alpha) = 1.

%dim = 2;
[ dim1 num1 ] = size(mu);
%num2 = length(kappa);
%num3 = length(alpha);

%if dim ~= dim1
%	error('dimensions are un-matched ...');
%end;
%if num1 ~= num2 || num1 ~= num3
%	error('the number of mixture components is un-matched ...');
%end;

prob = 0;
for ii = 1:num1
	prob = prob + alpha(ii) * vmf2_pdf(theta, phi, mu(:,ii), kappa(ii));
end;
