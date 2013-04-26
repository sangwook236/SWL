function prob = movm_pdf(dir_ang, mu, kappa, alpha)

% a mixture of von Mises distributions (1-dimensional)
%
% dir_ang: a direction angle, [rad].
% mu: mean direction angles, [rad].
% kappa: concentration parameters, kappa(i) >= 0.
% alpha: mixing coefficents, sum(alpha) = 1.

num1 = length(mu);
%num2 = length(kappa);
%num3 = length(alpha);

%if num1 ~= num2 || num1 ~= num3
%	error('the number of mixture components is un-matched ...');
%end;

prob = 0;
for ii = 1:num1
	prob = prob + alpha(ii) * vm_pdf(dir_ang, mu(ii), kappa(ii));
end;
