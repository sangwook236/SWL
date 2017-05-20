function [ perpendicular_dist1 perpendicular_dist2 cos_theta ] = distance_between_two_line_segments(ls1, ls2)
% The distance between two finite line segments, 1s1 & ls2.
% Each line segment: ls = [ x1 y1 x2 y2 ].
%	(x1, y1) - (x2, y2).
%
% [ perpendicular_dist1 perpendicular_dist2 cos_theta ] = distance_between_two_line_segments(ls1, ls2)
%	perpendicular_dist1: the perpendicular distance from the center of ls1 to the infinite line including ls2.
%	perpendicular_dist1: the perpendicular distance from the center of ls2 to the infinite line including ls1.
%	theta: the angle between the two line segments.
%	cos_theta: the cosine of theta.
%
% distance = perpendicular_dist1 / abs(cos_theta) + perpendicular_dist2 / abs(cos_theta).

x1a = ls1(:,1);
y1a = ls1(:,2);
x2a = ls1(:,3);
y2a = ls1(:,4);
x1b = ls2(:,1);
y1b = ls2(:,2);
x2b = ls2(:,3);
y2b = ls2(:,4);

if any(0 == (x2a - x1a) & 0 == (y2a - y1a)) || any(0 == (x2b - x1b) & 0 == (y2b - y1b))
	error('All finite line segments have finite lengths.');
	return;
end;

xc1 = (x1a + x2a) / 2;
yc1 = (y1a + y2a) / 2;

% The perpendicular distance between a point and a line.
perpendicular_dist = abs(a * xc1 + b * yc1 + c) / sqrt(a * a + b * b);

%
v1 = [b ; -a];
v2 = [x2 - x1 ; y2 - y1];
cos_theta = dot(v1, v2) / (norm(v1) * norm(v2));

return;
