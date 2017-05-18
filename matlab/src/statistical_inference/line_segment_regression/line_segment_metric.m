function [ dist cos_theta ] = line_segment_metric_with_outlier_removal(x1, y1, x2, y2, a, b, c)
% A distance between a finite line segment and an infinite line.
% A finite line segment: (x1, y1) - (x2, y2).
% An infinite line: a * x + b * y + c = 0.

xc = (x1 + x2) / 2;
yc = (y1 + y2) / 2;

% The perpendicular distance between a point and a line.
dist_perpendicular = abs(a * xc + b * yc + c) / sqrt(a*a + b*b);

if true
	v1 = [b ; -a];
	v2 = [x2 - x1 ; y2 - y1];
	cos_theta = dot(v1, v2) / (norm(v1) * norm(v2));
else
	angle1 = atan2(-a, b);
	angle2 = atan2(y2 - y1, x2 - x1);
	pi_2 = pi / 2;
	pi_3_4 = pi * 3 / 4;

	if angle1 > pi_2 & angle1 < pi_3_4
		angle1 = angle1 - pi;
	end;
	if angle2 > pi_2 & angle2 < pi_3_4
		angle2 = angle2 - pi;
	end;

	cos_theta = cos(angle1 - angle2);
end;

dist = dist_perpendicular / abs(cos_theta);
