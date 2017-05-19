function [ dist cos_theta ] = line_segment_metric(x1, y1, x2, y2, a, b, c, weight)
% A distance between a finite line segment and an infinite line.
%	A finite line segment: (x1, y1) - (x2, y2).
%	An infinite line: a * x + b * y + c = 0.
%
% [ perpendicular_dist cos_theta ] = line_segment_metric(x1, y1, x2, y2, a, b, c, weight_fun)
%	perpendicular_dist: the perpendicular distance from the center of the line segment to the line.
%	theta: the angle between the line segment and the line.
%	cos_theta: the cosine of theta.
% [ dist ] = line_segment_metric(x1, y1, x2, y2, a, b, c, weight_fun)
%	dist = perpendicular_dist / abs(cos_theta).
%	dist = perpendicular_dist / abs(weight_fun(theta)) if a weight function is given.

if 0 == (x2 - x1) && 0 == (y2 - y1)
	error('A finite line segment has a length.');
	return;
end;
if 0 == a && 0 == b
	error('Improper infinite line.');
	return;
end;

xc = (x1 + x2) / 2;
yc = (y1 + y2) / 2;

% The perpendicular distance between a point and a line.
dist = abs(a * xc + b * yc + c) / sqrt(a*a + b*b);

% Method 1.
v1 = [b ; -a];
v2 = [x2 - x1 ; y2 - y1];
cos_theta = dot(v1, v2) / (norm(v1) * norm(v2));

if nargout < 2
	if nargout >= 8
		dist = dist / abs(feval(weight_fun, acos(cos_theta)));
	else
		dist = dist / abs(cos_theta);
	end;
end;

return;

% Method 2 (not good).
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

if nargout < 2
	dist = dist / abs(cos_theta);
end;
