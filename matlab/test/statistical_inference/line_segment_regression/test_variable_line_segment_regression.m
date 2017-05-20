% Tests for line segment regression with variable lengths.

%addpath('D:/lib_repo/matlab/rnd/circstat-matlab_github');
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

% Sample.
num_instances = 100;
sigma_x = 20;
sigma_y = 5;
x_range = [-100 ; 100];

% Line segments: (x1, y1) - (x2, y2).
y_offset = normrnd(0, sigma, [num_instances, 1]);
x1 = x_range(1) + (x_range(2) - x_range(1)) .* rand([num_instances, 1]);
%x2 = x_range(1) + (x_range(2) - x_range(1)) .* rand([num_instances, 1]);
x2 = x1 + normrnd(0, sigma_x, [num_instances, 1]);
y1 = (-a / b) * x1 - (c / b) + normrnd(0, sigma_y, [num_instances, 1]);
y2 = (-a / b) * x2 - (c / b) + normrnd(0, sigma_y, [num_instances, 1]);

% Plot.
figure;
axis equal;
for ii = 1:length(x1)
	line([x1(ii) ; x2(ii)], [y1(ii) ; y2(ii)], 'Color', 'blue');
end;
line(x_range, (-a / b) * x_range - (c / b), 'Color', 'red');

% Subsegment.
ref_len = 2;
subsegment = generate_evenly_divided_subsegment([ x1 y1 x2 y2 ], ref_len);
%subsegment = generate_subsegment_randomly([ x1 y1 x2 y2 ], ref_len);
%subsegment = generate_centered_subsegment([ x1 y1 x2 y2 ], ref_len);

% FIXME [check] >> atan2 or atan?
%subsegment_angle = atan2(subsegment(:,4) - subsegment(:,2), subsegment(:,3) - subsegment(:,1));
subsegment_angle = atan((subsegment(:,4) - subsegment(:,2)) ./ (subsegment(:,3) - subsegment(:,1)));

% Optimize.
%options = optimoptions(@fminunc);
options = optimoptions(@fminunc, 'Algorithm', 'quasi-newton');
%options = optimoptions(@fminunc, 'Algorithm', 'trust-region', 'SpecifyObjectiveGradient', true);

% Starting guess.
angle_init_hat = circ_mean(subsegment_angle);
line_init = [tan(angle_init_hat) -1 0];

% NOTICE [decide] >> Decide whether outlier removal is included or not.
%	REF [function] >> line_segment_residual() & line_segment_weighted_residual().
%[line_hat, fval, exitflag, output] = fminunc(@(x) line_segment_residual([ x1 y1 x2 y2 ], line, pi / 2), line_init, options);
%[line_hat, fval, exitflag, output] = fminunc(@(x) line_segment_residual([ x1 y1 x2 y2 ], line, 80 * pi / 180), line_init, options);
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
