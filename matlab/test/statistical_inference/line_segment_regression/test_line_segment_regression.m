% Tests for line segment regression with the same length.

%addpath('D:/lib_repo/matlab/rnd/circstat-matlab_github');
%addpath('../../../src/statistical_inference/line_segment_regression');
%addpath('../../../src/topology');

% An infinite line: a * x + b * y + c = 0.
%a = 1;
%b = -1;
%c = 10;
a_range = [-10 ; 10];
b_range = [-10 ; 10];
c_range = [-10 ; 10];
a = a_range(1) + (a_range(2) - a_range(1)) * rand();
b = b_range(1) + (b_range(2) - b_range(1)) * rand();
c = c_range(1) + (c_range(2) - c_range(1)) * rand();

line_slope= atan(-a / b);
%line_slope= atan2(-a, b);
line_y_intercept = -c / b;

% Line segments: (x1, y1) - (x2, y2).
line_segment_length = 1;

% Sample.
%num_instances = 1000;
num_instances = 10;
sigma = 2;
kappa = 1;
x_range = [-100 ; 100];

line_segment_angle = circ_vmrnd(line_slope, kappa, [num_instances, 1]);
y_offset = normrnd(0, sigma, [num_instances, 1]);

xc = x_range(1) + (x_range(2) - x_range(1)) .* rand([num_instances, 1]);
yc = (-a / b) * xc - (c / b) + y_offset;

x1 = xc - line_segment_length * cos(line_segment_angle) / 2;
y1 = yc - line_segment_length * sin(line_segment_angle) / 2;
x2 = xc + line_segment_length * cos(line_segment_angle) / 2;
y2 = yc + line_segment_length * sin(line_segment_angle) / 2;

% Plot.
figure;
axis equal;
for ii = 1:length(x1)
	line([x1(ii) ; x2(ii)], [y1(ii) ; y2(ii)], 'Color', 'blue');
end;
line(x_range, (-a / b) * x_range - (c / b), 'Color', 'red');

% Optimize.
%options = optimoptions(@fminunc);
options = optimoptions(@fminunc, 'Algorithm', 'quasi-newton');
%options = optimoptions(@fminunc, 'Algorithm', 'trust-region', 'SpecifyObjectiveGradient', true);

% Starting guess.
%line_init = [4 5 1];  % Bad result: a / b and line_init(1) / line_init(2) should have the same sign.
%line_init = [1 -100 1];  % Bad result.
%line_init = [1 -5 1];
angle_init_hat = circ_mean(line_segment_angle);
line_init = [tan(angle_init_hat) -1 0];

% NOTICE [decide] >> Decide whether outlier removal is included or not.
%	REF [function] >> line_segment_residual() & line_segment_weighted_residual().
%[line_hat, fval, exitflag, output] = fminunc(@(line) line_segment_residual([ x1 y1 x2 y2 ], line, pi / 2), line_init, options);
%[line_hat, fval, exitflag, output] = fminunc(@(line) line_segment_residual([ x1 y1 x2 y2 ], line, 80 * pi / 180), line_init, options);
weight_fun = inline('scale * cos(x*2) - scale + 1', 'x', 'scale');  % 0 < scale <= 0.5.
weight_scale = 0.5 * 0.99;
%[line_hat, fval, exitflag, output] = fminunc(@(line) line_segment_weighted_residual([ x1 y1 x2 y2 ], line, @(x) weight_fun(x, weight_scale), pi / 2), line_init, options);
[line_hat, fval, exitflag, output] = fminunc(@(line) line_segment_weighted_residual([ x1 y1 x2 y2 ], line, @(x) weight_fun(x, weight_scale), 80 * pi / 180), line_init, options);

% Output the result.
line(x_range, (-line_hat(1) / line_hat(2)) * x_range - (line_hat(3) / line_hat(2)), 'Color', 'green');
disp(sprintf('Exit flag: %d', exitflag));

% True line.
[ a b c ] / b
% Estimated line.
line_hat / line_hat(2)
