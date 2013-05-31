function logprob = vm_conjugate_log_prior_pdf(mu, kappa, mu_0, R_0, c)

% the conjugate prior of von Mises distribution (1-dimensional)
%
% mu: a mean direction angle, [0 2*pi), [rad].
% kappa: a concentration parameter, kappa >= 0.
% mu_0: the prior density paramter of the mean direction angle, [0 2*pi), [rad].
% R_0: the prior density paramter of the resultant length, R_0 > 0.
% c: the prior density paramter.

% [ref] "Finding the Location of a Signal: A Bayesian Analysis", P. Guttorp and R. A .Lockhart, JASA, 1988.
% [ref] "A Bayesian Analysis of Directional Data Using the von Mises-Fisher Distribution", G. Nunez-Antonio and E. Gutierrez-Pena, CSSC, 2005.

logprob = log(vm_conjugate_prior_pdf(mu, kappa, mu_0, R_0, c));
