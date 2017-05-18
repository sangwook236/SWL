%addpath('D:/lib_repo/matlab/rnd/circstat-matlab_github');
%addpath('../../../src/topology');

% Infinite line: a * x + b * y + c = 0.
%a = 1;
%b = -1;
%c = 10;
a_range = [-10 ; 10];
b_range = [-10 ; 10];
c_range = [-10 ; 10];
a = a_range(1) + (a_range(2) - a_range(1)) * rand();
b = b_range(1) + (b_range(2) - b_range(1)) * rand();
c = c_range(1) + (c_range(2) - c_range(1)) * rand();

theta_line = atan(-a / b);
%theta_line = atan2(-a, b);
y_intercept_line = -c / b;

% Line segment: (x1, y1) - (x2, y2).
len_segm = 1;

% Sample.
num_instances = 1000;
sigma = 2;
kappa = 1;
x_range = [-100 ; 100];

angle = circ_vmrnd(theta_line, kappa, [num_instances, 1]);
y_offset = normrnd(0, sigma, [num_instances, 1]);
x_c = x_range(1) + (x_range(2) - x_range(1)) .* rand([num_instances, 1]);
y_c = (-a / b) * x_c - (c / b) + y_offset;

x1 = x_c - len_segm * cos(angle) / 2;
y1 = y_c - len_segm * sin(angle) / 2;
x2 = x_c + len_segm * cos(angle) / 2;
y2 = y_c + len_segm * sin(angle) / 2;

% Plot.
figure;
axis equal;
line(x_range, (-a / b) * x_range - (c / b), 'Color', 'red');
for ii = 1:length(x1)
	line([x1(ii) ; x2(ii)], [y1(ii) ; y2(ii)], 'Color', 'blue');
end;

% Optimize.
options = optimoptions(@fminunc);
options = optimoptions(@fminunc, 'Algorithm', 'quasi-newton');
%options = optimoptions(@fminunc, 'Algorithm', 'trust-region', 'SpecifyObjectiveGradient', true);

angle_threshold = pi / 4;
% Starting guess.
%x_init = [4 ; 5 ; 1];  % Bad result: a / b and x_init(1) / x_init(2) should have the same sign.
%x_init = [1 ; -100 ; 1];  % Bad result.
%x_init = [1 ; -5 ; 1];
angle_hat = circ_mean(angle);
x_init = [tan(angle_hat) ; -1 ; 0];

% NOTICE [decide] >> Decide whether outlier removal is included or not.
%	REF [function] >> line_segment_residual().
[x, fval, exitflag, output] = fminunc(@(x) line_segment_residual(x, x1, y1, x2, y2, angle_threshold), x_init, options);

% Output the result.
line(x_range, (-x(1) / x(2)) * x_range - (x(3) / x(2)), 'Color', 'green');
disp(sprintf('Exit flag: %d', exitflag));
