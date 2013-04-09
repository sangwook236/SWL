function [ est_mu, est_sigma, est_alpha, step ] = em_MoG(X, init_mu, init_sigma, init_alpha, max_step, tol)

% expectation-maximization (EM) algorithm for mixtures of Gaussian distributions (MoG)
% 1-dimensional

num1 = length(init_mu);
num2 = length(init_sigma);
num3 = length(init_alpha);
if num1 == num2 && num1 == num3
	num_mix_comp = num1;
else
	error('dimensions of inputs are not matched ...');
end;

est_mu = init_mu;
est_sigma = init_sigma;
est_alpha = init_alpha;

num_sample = length(X);

gamma = zeros(num_sample, num_mix_comp);
prob_comp = zeros(1, num_mix_comp);
looping = true;
step = 0;
while looping && step <= max_step
	% E-step
    for nn = 1:num_sample
		for kk = 1:num_mix_comp
	        prob_comp(kk) = normpdf(X(nn), est_mu(kk), est_sigma(kk));
		end;
       	gamma(nn,:) = (est_alpha .* prob_comp) / dot(est_alpha, prob_comp);
    end;

    est_mu_old = est_mu;
    est_sigma_old = est_sigma;
    est_alpha_old = est_alpha;

	% M-step
	sum_gamma = sum(gamma);
	est_alpha = sum_gamma / sum(sum_gamma);
    for kk = 1:num_mix_comp
        est_mu(kk) = dot(gamma(:,kk), X) / sum_gamma(kk);
        est_sigma(kk) = sqrt(dot(gamma(:,kk), (X - est_mu(kk)) .* (X - est_mu(kk))) / sum_gamma(kk));
    end;

	looping = sum(abs(est_mu - est_mu_old) > tol) > 0 || ...
		sum(abs(est_sigma - est_sigma_old) > tol) > 0 || ...
		sum(abs(est_alpha - est_alpha_old) > tol) > 0;

    step = step + 1;
    if rem(step, 100) == 0
    	sprintf('iteration #%d', step)
    end;
end;
