function logprob = log_vm_pdf(dir, mean_dir, kappa)

% log probability of von Mises distribution (1-dimensional)
%
% dir: a direction angle, [0 2*pi), [rad].
% mean_dir: a mean direction angle, [rad].
% kappa: a concentration parameter, kappa >= 0.

logprob = kappa * cos(dir - mean_dir) - log(2 * pi * besseli(0, kappa));
