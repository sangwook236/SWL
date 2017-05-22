%addpath('../../src/topology');

% Distance between two finite line segments.
ls1 = [
	0 0 1 0
	0 0 1 0
	0 0 1 0
	0 0 1 0
	%0 0 1 0
];
%ls1 = repmat([0 0 1 0], [4 1]);
ls2 = [
	0 1 1 1
	[ 0.5 1 ] - 0.5 * [ cos(pi/6) sin(pi/6) ] [ 0.5 1 ] + 0.5 * [ cos(pi/6) sin(pi/6) ]
	[ 0.5 1 ] - 0.5 * [ cos(pi/4) sin(pi/4) ] [ 0.5 1 ] + 0.5 * [ cos(pi/4) sin(pi/4) ]
	[ 0.5 1 ] - 0.5 * [ cos(pi/3) sin(pi/3) ] [ 0.5 1 ] + 0.5 * [ cos(pi/3) sin(pi/3) ]
	%[ 0.5 1 ] - 0.5 * [ cos(pi/2) sin(pi/2) ] [ 0.5 1 ] + 0.5 * [ cos(pi/2) sin(pi/2) ]
];

[ dist_perp1 dist_perp2 cos_theta ] = distance_between_two_line_segments(ls1, ls2);

dist = (dist_perp1 + dist_perp2) ./ abs(2 * cos_theta)
