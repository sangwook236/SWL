function [ est_mean_dir, est_kappa, est_alpha, step ] = em_MovM(X, init_mean_dir, init_kappa, init_alpha, max_step, tol)

% expectation-maximization (EM) algorithm for mixtures of von Mises distributions (MovM)

num1 = length(init_mean_dir);
num2 = length(init_kappa);
num3 = length(init_alpha);
if num1 == num2 && num1 == num3
	num_mix_comp = num1;
else
	error('dimensions of inputs are not matched ...');
end;

est_mean_dir = init_mean_dir;
est_kappa = init_kappa;
est_alpha = init_alpha;

num_sample = length(X);

sinX = sin(X);
cosX = cos(X);

gamma = zeros(num_sample, num_mix_comp);
prob_comp = zeros(1, num_mix_comp);
looping = true;
step = 0;
while looping && step <= max_step
	% E-step
    for nn = 1:num_sample
		for kk = 1:num_mix_comp
	        prob_comp(kk) = circ_vmpdf(X(nn), est_mean_dir(kk), est_kappa(kk));
		end;
       	gamma(nn,:) = (est_alpha .* prob_comp) / dot(est_alpha, prob_comp);
    end;

    est_mean_dir_old = est_mean_dir;
    est_kappa_old = est_kappa;
    est_alpha_old = est_alpha;

	% M-step
	sum_gamma = sum(gamma);
	est_alpha = sum_gamma / sum(sum_gamma);
    for kk = 1:num_mix_comp
        est_mean_dir(kk) = pi + atan2(dot(gamma(:,kk), sinX), dot(gamma(:,kk), cosX));

        A = dot(gamma(:,kk), cos(X - est_mean_dir(kk))) / sum_gamma(kk);
        est_kappa(kk) = fzero(@(kappa) A - besseli(1, kappa) / besseli(0, kappa), est_kappa(kk));
    end;

	looping = sum(abs(est_mean_dir - est_mean_dir_old) > tol) > 0 || ...
		sum(abs(est_kappa - est_kappa_old) > tol) > 0 || ...
		sum(abs(est_alpha - est_alpha_old) > tol) > 0;

    step = step + 1;
    if rem(step, 100) == 0
    	sprintf('iteration #%d', step)
    end;
end;

for kk = 1:num_mix_comp
	if est_kappa(kk) < 0
		est_kappa(kk) = -est_kappa(kk);
		est_mean_dir(kk) = est_mean_dir(kk) + pi;
		if est_mean_dir(kk) > 2*pi
			est_mean_dir(kk) = est_mean_dir(kk) - 2*pi;
		end;
	end;
end;
