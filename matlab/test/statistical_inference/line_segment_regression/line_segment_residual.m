function resid = line_segment_residual(x, x1, y1, x2, y2, angle_threshold)
% A finite line segment: (x1, y1) - (x2, y2).
% An infinite line: a * x + b * y + c = 0.
%	a = x(1), b = x(2), c = x(3).

abs_cos_angle_threshold = abs(cos(angle_threshold));
inlier_flag = zeros(size(x1));

resid = 0;
for ii = 1:length(x1)
	[ dist cos_theta ] = line_segment_metric(x1(ii), y1(ii), x2(ii), y2(ii), x(1), x(2), x(3));

	% Hanle outliers.
	inlier_flag(ii) = abs(cos_theta) > abs_cos_angle_threshold;
	if inlier_flag(ii)
		resid = resid + dist;
	end;
end;

% Print the number of inliers.
%sum(inlier_flag(:) > 0)
