function prob = vm_conjugate_prior_pdf(dir, mean_dir, kappa)

% the conjugate prior of von Mises distribution (1-dimensional)
%
% dir: a direction angle, [0 2*pi), [rad].
% mean_dir: a mean direction angle, [rad].
% kappa: a concentration parameter, kappa >= 0.

% target distribution
R_n = 20;
theta_n = pi;
c = 5;
n = 100;
kappa0 = 1;
f = @(theta) exp(kappa0 * R_n * cos(theta - theta_n)) / besseli(0, kappa0)^(c + n);

num_samples = 10000;
burn_in_period = 1000;
thinning_period = 5;
smpl5 = slicesample(1, num_samples, 'pdf', f, 'thin', thinning_period, 'burnin', burn_in_period);
smpl5 = mod(smpl5, 2*pi);

prob = exp(kappa * cos(dir - mean_dir)) / (2 * pi * besseli(0, kappa));
