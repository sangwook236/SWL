function [ resid inlier_flag ] = line_segment_weighted_residual(x, x1, y1, x2, y2, weight_fun, angle_threshold)
% A finite line segment: (x1, y1) - (x2, y2).
% An infinite line: a * x + b * y + c = 0.
%	a = x(1), b = x(2), c = x(3).

if nargin >= 7
	outlier_removal = true;
end;

abs_cos_angle_threshold = abs(cos(angle_threshold));
inlier_flag = zeros(size(x1));

resid = 0;
for ii = 1:length(x1)
	[ dist_perp cos_theta ] = line_segment_metric(x1(ii), y1(ii), x2(ii), y2(ii), x(1), x(2), x(3), weight_fun);

	abs_cos_theta = abs(cos_theta);

	% Check inliers.
	if outlier_removal
		inlier_flag(ii) = abs_cos_theta > abs_cos_angle_threshold;
		if inlier_flag(ii)
			resid = resid + dist_perp / abs_cos_theta;
		end;
	else
		resid = resid + dist_perp / abs_cos_theta;
	end;
end;

% Print the number of inliers.
%sum(inlier_flag(:) > 0)
