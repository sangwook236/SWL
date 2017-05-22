%addpath('../../src/topology');

% Distance between a finite line segment and an infinite line.
line = [
	0 -1 0
	0 -1 0
	0 -1 0
	0 -1 0
	%0 -1 0
];
%line = repmat([0 -1 0], [4 1]);
ls = [
	0 1 1 1
	[ 0.5 1 ] - 0.5 * [ cos(pi/6) sin(pi/6) ] [ 0.5 1 ] + 0.5 * [ cos(pi/6) sin(pi/6) ]
	[ 0.5 1 ] - 0.5 * [ cos(pi/4) sin(pi/4) ] [ 0.5 1 ] + 0.5 * [ cos(pi/4) sin(pi/4) ]
	[ 0.5 1 ] - 0.5 * [ cos(pi/3) sin(pi/3) ] [ 0.5 1 ] + 0.5 * [ cos(pi/3) sin(pi/3) ]
	%[ 0.5 1 ] - 0.5 * [ cos(pi/2) sin(pi/2) ] [ 0.5 1 ] + 0.5 * [ cos(pi/2) sin(pi/2) ]
];

[ dist_perp cos_theta ] = distance_between_line_segment_and_infinite_line(ls, line);

dist = dist_perp ./ abs(cos_theta)
