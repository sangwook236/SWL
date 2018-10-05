function prob = mog_pdf(x, mu, sigma, alpha)

% a mixture of Gaussian distributions (1-dimensional)


num1 = length(mu);
%num2 = length(sigma);
%num3 = length(alpha);

%if num1 ~= num2 || num1 ~= num3
%	error('the number of mixture components is un-matched ...');
%end;

prob = 0;
for ii = 1:num1
	prob = prob + alpha(ii) * normpdf(x, mu(ii), sigma(ii));
end;
