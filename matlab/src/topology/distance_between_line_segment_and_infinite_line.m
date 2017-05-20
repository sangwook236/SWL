function [ perpendicular_dist cos_theta ] = distance_between_line_segment_and_infinite_line(ls, line)
% The distance between a finite line segment and an infinite line.
% A finite line segment: ls = [ x1 y1 x2 y2 ].
%	(x1, y1) - (x2, y2).
% An infinite line: line = [ a b c ].
%	a * x + b * y + c = 0.
%
% [ perpendicular_dist cos_theta ] = distance_between_line_segment_and_infinite_line(ls, line)
%	perpendicular_dist: the perpendicular distance from the center of the line segment to the line.
%	theta: the angle between the line segment and the line.
%	cos_theta: the cosine of theta.
%
% distance = perpendicular_dist / abs(cos_theta).

x1 = ls(:,1);
y1 = ls(:,2);
x2 = ls(:,3);
y2 = ls(:,4);
a = line(1);
b = line(2);
c = line(3);

if any(0 == (x2 - x1) & 0 == (y2 - y1))
	error('A finite line segment has a finite length.');
	return;
end;
if 0 == a && 0 == b
	error('Improper infinite line.');
	return;
end;

xc = (x1 + x2) / 2;
yc = (y1 + y2) / 2;

% The perpendicular distance between a point and a line.
perpendicular_dist = abs(a * xc + b * yc + c) / sqrt(a * a + b * b);

%
v1 = [b -a];
v2 = [x2 - x1 y2 - y1];
norm1 = norm(v1);
num_ls = size(v2, 1);
cos_theta = zeros(num_ls, 1);
for ii = 1:num_ls
	cos_theta(ii) = dot(v1, v2(ii,:)) / (norm1 * norm(v2(ii,:)));
end;

return;
