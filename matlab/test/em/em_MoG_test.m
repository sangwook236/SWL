%---------------------------------------------------------------
%addpath('../../src/em');
%addpath('../../src/statistics');

%---------------------------------------------------------------
num_mix_comp = 3;
mu_true = [ 0 6 10 ];
sigma_true = [ 1.1 0.9 1.5 ];
alpha_true = [ 0.3 0.4 0.3 ];

%---------------------------------------------------------------
% generate sample

num_sample = 1000;
num_comp_sample = round(alpha_true * num_sample);
num_sample = sum(num_comp_sample);

X = zeros(num_sample,1);
start_idx = 1;
end_idx = 0;
for ii = 1:num_mix_comp
	end_idx = end_idx + num_comp_sample(ii);
	X(start_idx:end_idx) = normrnd(mu_true(ii), sigma_true(ii), [num_comp_sample(ii), 1]);
	start_idx = end_idx + 1;
end;

mu_init = [ 3 8 12 ];
%mu_init = [ 1 2 3 ];
sigma_init = [ 1 1 1 ];
alpha_init = [ 1/3 1/3 1/3 ];

%---------------------------------------------------------------
% expectation-maximization (EM) algorithm for mixtures of Gaussian distributions (MoG)
disp('batch EM ...');

max_step = 1000;
tol = 1e-5;

[ mu_est, sigma_est, alpha_est, step ] = em_MoG(X, mu_init, sigma_init, alpha_init, max_step, tol);

%---------------------------------------------------------------
% display results
mu_est
sigma_est
alpha_est
step

figure;
axis_range = [ -5 15 ];
subplot(3,1,1), hist(X, 100);
axis_val = axis;
axis([ axis_range(1) axis_range(2) axis_val(3) axis_val(4) ]);
subplot(3,1,2), ezplot(@(x) mog_pdf(x, mu_true, sigma_true, alpha_true), axis_range);
subplot(3,1,3), ezplot(@(x) mog_pdf(x, mu_est, sigma_est, alpha_est), axis_range);
