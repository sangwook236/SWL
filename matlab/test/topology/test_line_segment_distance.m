%addpath('../../src/topology');

% Distances between an infinite line and a finite line segment.

Ls = 5;  % The length of line segments

% Dp: the perpendicular distance of the center of the line segment onto the line.
Dp = linspace(0, 100, 1001);
% Ar: the angle between the line segment and the line.
%Ar = linspace(-pi/2, pi/2, 1001);
Ar = linspace(-pi, pi, 1001);
[gridDp, gridAr] = meshgrid(Dp, Ar);

%-----------------------------------------------------------
% Distance = midpoint distance.
% REF [paper] >> "Evaluation of Established Line Segment Distance Functions", PRIA 2016.
%
% Result: Bad.

distDp = 5 * Dp.^2;
distAr = sin(Ar).^2 * Ls^2 / 2;

figure;
subplot(2,1,1);
plot(Dp, distDp);
subplot(2,1,2);
plot(Ar, distAr);

distDpAr = 5 * gridDp.^2 + sin(gridAr).^2 * Ls^2 / 2;

figure;
mesh(gridDp, gridAr, distDpAr);

%-----------------------------------------------------------
% A weighted distance between a finite line segment and an infinite line.
%	distnace = a perpendicular distance weighted by projection length.
%	dist = Dp / cos(Ar).
%
% Result: Good, but go to infinity around pi / 2.
% REF [file] >> ${SWL_HOME}/matlab/src/topology/distance_between_line_segment_and_infinite_line.m.

distAr = abs(Dp(100) ./ cos(gridAr));
figure;
plot(Ar, distAr);
ax = axis;
axis([ax(1), ax(2), ax(3), 2000]);

dist = abs(gridDp ./ cos(gridAr));

figure;
mesh(gridDp, gridAr, dist);
ax = axis;
%axis([ax(1), ax(2), ax(3), ax(4), ax(5), 2000]);

%-----------------------------------------------------------

% A weighted distance between a finite line segment and an infinite line.
%	dist = perpendicular_dist / abs(weight).
%
% Result: Good, finite and differentiable around pi / 2.
% REF [file] >> ${SWL_HOME}/matlab/src/topology/distance_between_line_segment_and_infinite_line.m.

% Weight function.
%weight = inline('cos(x)', 'x');
weight_fun = inline('scale * cos(x*2) - scale + 1', 'x', 'scale');  % 0 < scale <= 0.5.
weight_scale = 0.5 * 0.99;

%ezplot(@(x) weight_fun(x, weight_scale))

dist = abs(gridDp ./ weight_fun(gridAr, weight_scale));

figure;
mesh(gridDp, gridAr, dist);
%surf(gridDp, gridAr, dist);
ax = axis;
%axis([ax(1), ax(2), ax(3), ax(4), ax(5), 2000]);
