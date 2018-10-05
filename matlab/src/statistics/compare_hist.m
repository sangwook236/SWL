function distance = compare_hist(H1, H2, method)

N = length(H1);

if N ~= length(H2)
	error('sizes of two histograms have to be the same');
end;

eps = 1e-5;
switch lower(method)
	case { 'correlation', 'correl' }
		H1 = H1 - mean(H1);
		H2 = H2 - mean(H2);
		distance = dot(H1, H2) / sqrt(sum(H1.^2) * sum(H2.^2));
	case { 'chisquare', 'chisqr' }
		%distance = sum((H1 - H2).^2 ./ H1);
		distance = 0;
		for ii = 1:N
			if H1(ii) > eps
				distance = distance + (H1(ii) - H2(ii))^2 / H1(ii);
			end;
		end;
	case { 'intersection', 'intersect' }
		distance = sum(min(H1, H2));
	case { 'bhattacharyya', 'bhatta' }
		distance = sqrt(1 - sum(sqrt(H1 .* H2)) / (N * sqrt(mean(H1) * mean(H2))));
	otherwise
		distance = inf;
		error('unknown method.')
end
