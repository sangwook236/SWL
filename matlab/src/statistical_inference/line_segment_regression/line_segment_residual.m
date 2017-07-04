function [ resid inlier_flag ] = line_segment_residual(ls, ls_weight, line, angle_threshold)
% A finite line segment: ls = [ x1 y1 x2 y2 ].
%	(x1, y1) - (x2, y2).
% An infinite line: line = [ a b c ].
%	a * x + b * y + c = 0.
% angle_threshold: if the angle between a line segment and the infinite line is larger than angle_threshold, the line segment is excluded from regression.

abs_cos_angle_threshold = abs(cos(angle_threshold));
num_ls = size(ls, 1);
inlier_flag = zeros(num_ls, 1);

[ dist_perp cos_theta ] = distance_between_line_segment_and_infinite_line(ls, line);
abs_cos_theta = abs(cos_theta);

% Check inliers.
if nargin >= 4
	inlier_flag = abs_cos_theta > abs_cos_angle_threshold;
	if isempty(ls_weight)
		resid = sum((dist_perp ./ abs_cos_theta) .* inlier_flag);
	else
		resid = sum(ls_weight .* (dist_perp ./ abs_cos_theta) .* inlier_flag);
	end;
else
	if isempty(ls_weight)
		resid = sum(dist_perp ./ abs_cos_theta);
	else
		resid = sum(ls_weight .* dist_perp ./ abs_cos_theta);
	end;
end;

% Print the number of inliers.
%sum(inlier_flag(:) > 0)
