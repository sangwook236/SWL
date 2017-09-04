function uni = find_unique(X, tol)

if nargin < 2
	tol = eps;
end;

if isvector(X)
	kk = 1;
	len = length(X);
	for ii = 1:len
		is_unique = true;
		for jj = 1:ii-1
		%for jj = (ii+1):len
			if abs(X(ii) - X(jj)) <= tol
				is_unique = false;
				break;
			end;
		end;
		if is_unique
			uni(kk) = X(ii);
			kk = kk + 1;
		end;
	end;
else
	% Column-wise comparison.

	kk = 1;
	len = size(X, 2);
	for ii = 1:len
		is_unique = true;
		for jj = 1:ii-1
		%for jj = (ii+1):len
			if abs(X(:,ii) - X(:,jj)) <= tol
				is_unique = false;
				break;
			end;
		end;
		if is_unique
			uni(:,kk) = X(:,ii);
			kk = kk + 1;
		end;
	end;
end;
