function [ perpendicular_dist1 perpendicular_dist2 cos_theta ] = distance_between_two_line_segments(ls1, ls2)
% The distance between two finite line segments, 1s1 & ls2.
% Each line segment: ls = [ x1 y1 x2 y2 ].
%	(x1, y1) - (x2, y2).
%
% [ perpendicular_dist1 perpendicular_dist2 cos_theta ] = distance_between_two_line_segments(ls1, ls2)
%	perpendicular_dist1: the perpendicular distance from the center of ls2 to the infinite line including ls1.
%	perpendicular_dist2: the perpendicular distance from the center of ls1 to the infinite line including ls2.
%	theta: the angle between the two line segments.
%	cos_theta: the cosine of theta.
%
% distance = (perpendicular_dist1 + perpendicular_dist2) / abs(2 * cos_theta).

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
xc2 = (x1b + x2b) / 2;
yc2 = (y1b + y2b) / 2;

% Infinite line equation.
a1 = y2a - y1a;
b1 = x1a - x2a;
c1 = y1a .* (x2a - x1a) - x1a .* (y2a - y1a);
a2 = y2b - y1b;
b2 = x1b - x2b;
c2 = y1b .* (x2b - x1b) - x1b .* (y2b - y1b);

% The perpendicular distance between a point and a line.
perpendicular_dist1 = abs(a1 .* xc2 + b1 .* yc2 + c1) ./ sqrt(a1.^2 + b1.^2);
perpendicular_dist2 = abs(a2 .* xc1 + b2 .* yc1 + c2) ./ sqrt(a2.^2 + b2.^2);

% The angle between the two line segments.
v1 = [-b1 a1];
v2 = [-b2 a2];
num_ls = size(v1, 1);
cos_theta = zeros(num_ls, 1);
for ii = 1:num_ls
	cos_theta(ii) = dot(v1(ii,:), v2(ii,:)) / (norm(v1(ii,:)) * norm(v2(ii,:)));
end;

return;
