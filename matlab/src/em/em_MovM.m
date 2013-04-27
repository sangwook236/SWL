function [ est_mu, est_kappa, est_alpha, step ] = em_MovM(X, num_clusters, init_mu, init_kappa, init_alpha, max_step, tol)

% expectation-maximization (EM) algorithm for mixtures of von Mises distributions (MovM)

init_mu_rng = [ 0 2*pi ];
init_kappa_rng = [ 0.5 5 ];
est_kappa_rng = [ -700 700 ];
% random initialization
if isempty(init_mu)
    init_mu = init_mu_rng(1) + (init_mu_rng(2) - init_mu_rng(1)) * rand(1, num_clusters);
end;
if isempty(init_kappa)
    init_kappa = init_kappa_rng(1) + (init_kappa_rng(2) - init_kappa_rng(1)) * rand(1, num_clusters);
end;
if isempty(init_alpha)
    while true
        init_alpha = rand(1, num_clusters);
        if sum(init_alpha) > 1e-3
            init_alpha = init_alpha / sum(init_alpha);
            break;
        end;
    end;
end;

num1 = length(init_mu);
num2 = length(init_kappa);
num3 = length(init_alpha);
if num1 ~= num_clusters || num2 ~= num_clusters || num3 ~= num_clusters
	error('dimensions of inputs are not matched ...');
end;

est_mu = init_mu;
est_kappa = init_kappa;
est_alpha = init_alpha;

num_sample = length(X);

sinX = sin(X);
cosX = cos(X);

gamma = zeros(num_sample, num_clusters);
prob_comp = zeros(1, num_clusters);
looping = true;
step = 0;
fzero_options = optimset('fzero');
%fzero_options.Display = 'off';
%fzero_options.FunValCheck = 'off';
%fzero_options.TolX = tol;
fzero_options = optimset(fzero_options, 'Display', 'off', 'FunValCheck', 'off', 'TolX', tol);
while looping && step <= max_step
	% E-step
    for nn = 1:num_sample
		for kk = 1:num_clusters
	        prob_comp(kk) = circ_vmpdf(X(nn), est_mu(kk), est_kappa(kk));
		end;
       	gamma(nn,:) = (est_alpha .* prob_comp) / dot(est_alpha, prob_comp);
    end;

    est_mean_dir_old = est_mu;
    est_kappa_old = est_kappa;
    est_alpha_old = est_alpha;

	% M-step
	sum_gamma = sum(gamma);
	est_alpha = sum_gamma / sum(sum_gamma);
    for kk = 1:num_clusters
        est_mu(kk) = pi + atan2(dot(gamma(:,kk), sinX), dot(gamma(:,kk), cosX));

        A = dot(gamma(:,kk), cos(X - est_mu(kk))) / sum_gamma(kk);
    	if A < -0.995
    		est_kappa(kk) = est_kappa_rng(1);
    	elseif A > 0.995
    		est_kappa(kk) = est_kappa_rng(2);
    	else
    		while true
		        [ est_kappa(kk), fval, exitflag ] = fzero(@(kappa) A - besseli(1, kappa) / besseli(0, kappa), est_kappa(kk), fzero_options);
		        %sprintf('kk: %d, kappa: %f, A: %f, fval: %f, exitflag: %d', kk, est_kappa(kk), A, fval, exitflag)

		        if 1 ~= exitflag || isnan(est_kappa(kk))
		        	sprintf('fzero''s exitflag: %d, A: %f, est-kappa: %f', exitflag, A, est_kappa(kk))

					if est_kappa(kk) < est_kappa_rng(1) || est_kappa(kk) > est_kappa_rng(2)
						est_kappa(kk) = est_kappa_rng(1) + (est_kappa_rng(2) - est_kappa_rng(1)) .* rand;
					end;
		        else
		        	break;
		        end;
		    end;
    	end;
    end;

	looping = sum(abs(est_mu - est_mean_dir_old) > tol) > 0 || ...
		sum(abs(est_kappa - est_kappa_old) > tol) > 0 || ...
		sum(abs(est_alpha - est_alpha_old) > tol) > 0;

    step = step + 1;
    if rem(step, 100) == 0
    	sprintf('iteration #%d', step)
    end;
end;

for kk = 1:num_clusters
	if est_kappa(kk) < 0
		est_kappa(kk) = -est_kappa(kk);
		est_mu(kk) = est_mu(kk) + pi;
		if est_mu(kk) > 2*pi
			est_mu(kk) = est_mu(kk) - 2*pi;
		end;
	end;
end;
