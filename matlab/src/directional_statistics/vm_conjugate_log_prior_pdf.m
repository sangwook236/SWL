function logprob = log_vm_conjugate_prior_pdf(dir, mean_dir, kappa)

% log probability of the conjugate prior of von Mises distribution (1-dimensional)
%
% dir: a direction angle, [0 2*pi), [rad].
% mean_dir: a mean direction angle, [rad].
% kappa: a concentration parameter, kappa >= 0.

logprob = log(vm_conjugate_prior_pdf(dir, mean_dir, kappa));
