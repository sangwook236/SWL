function [ resid inlier_flag ] = line_segment_weighted_residual(ls, ls_weight, line, weight_fun, angle_threshold)
% A finite line segment: ls = [ x1 y1 x2 y2 ].
%	(x1, y1) - (x2, y2).
% An infinite line: line = [ a b c ].
%	a * x + b * y + c = 0.
% weight_fun:
%	Use a distance (= perpendicular_dist / abs(weight_fun(theta))) instead of the original distance (= perpendicular_dist / abs(cos(theta))).
% angle_threshold: if the angle between a line segment and the infinite line is larger than angle_threshold, the line segment is excluded from regression.

abs_cos_angle_threshold = abs(cos(angle_threshold));
num_ls = size(ls, 1);
inlier_flag = zeros(num_ls, 1);

[ dist_perp cos_theta ] = distance_between_line_segment_and_infinite_line(ls, line);
abs_wf_theta = abs(feval(weight_fun, acos(cos_theta)));

% Check inliers.
if nargin >= 5
	inlier_flag = abs_wf_theta > abs_cos_angle_threshold;
	if isempty(ls_weight)
		resid = sum((dist_perp ./ abs_wf_theta) .* inlier_flag);
	else
		resid = sum(ls_weight .* (dist_perp ./ abs_wf_theta) .* inlier_flag);
	end;
else
	if isempty(ls_weight)
		resid = sum(dist_perp ./ abs_wf_theta);
	else
		resid = sum(ls_weight .* dist_perp ./ abs_wf_theta);
	end;
end;

% Print the number of inliers.
%sum(inlier_flag(:) > 0)
