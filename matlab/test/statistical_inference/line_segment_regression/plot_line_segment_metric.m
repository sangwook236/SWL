% Metrics between an infinite line and a finite line segment.

Ls = 5;  % The length of line segments

% Dp: the perpendicular distance of the center of the line segment onto the line.
Dp = linspace(0, 100, 1001);
% Ar: the angle between the line segment and the line.
Ar = linspace(0, pi, 1001);
[gridDp, gridAr] = meshgrid(Dp, Ar);

%-----------------------------------------------------------
% Metric = midpoint distance.
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
% Metric = orthogonal distance divided by projection length.
% dist = Dp / cos(Ar).
%
% Result: ???.

distAr = abs(Dp(100) ./ cos(gridAr));
figure;
plot(Ar, distAr);
ax = axis;
axis([ax(1), ax(2), ax(3), 2000]);

dist = abs(gridDp ./ cos(gridAr));

figure;
mesh(gridDp, gridAr, dist);
ax = axis;
axis([ax(1), ax(2), ax(3), ax(4), ax(5), 2000]);
