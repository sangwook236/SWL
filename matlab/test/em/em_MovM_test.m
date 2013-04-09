%---------------------------------------------------------------
%addpath('D:\work_center\sw_dev\matlab\rnd\src\directional_statistics\circstat\CircStat2012a');
%addpath('../../src/em');
%addpath('../../src/directional_statistics');

%---------------------------------------------------------------
num_mix_comp = 3;
mu_true = [ 1 2.5 5 ];
kappa_true = [ 4 5 2 ];
alpha_true = [ 0.25 0.25 0.5 ];

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
	X(start_idx:end_idx) = circ_vmrnd(mu_true(ii), kappa_true(ii), num_comp_sample(ii));
	start_idx = end_idx + 1;
end;
for ii = 1:num_sample
	if X(ii) < 0
		X(ii) = X(ii) + 2*pi;
	end;
end;

init_mean_dir = [ pi/6 pi*5/6 pi*9/6 ];
init_kappa = [ 1 1 1 ];
init_alpha = [ 1/3 1/3 1/3 ];

%---------------------------------------------------------------
% expectation-maximization (EM) algorithm for mixtures of von Mises distributions (MovM)
disp('batch EM ...');

max_step = 1000;
tol = 1e-5;

[ mu_est, kappa_est, alpha_est, step ] = em_MovM(X, init_mean_dir, init_kappa, init_alpha, max_step, tol);

%---------------------------------------------------------------
% display results
mu_est
kappa_est
alpha_est
step

figure;
axis_range = [ 0 2*pi ];
subplot(3,1,1), hist(X, 100);
axis_val = axis;
axis([ axis_range(1) axis_range(2) axis_val(3) axis_val(4) ]);
subplot(3,1,2), ezplot(@(x) movm_pdf(x, mu_true, kappa_true, alpha_true), axis_range);
subplot(3,1,3), ezplot(@(x) movm_pdf(x, mu_est, kappa_est, alpha_est), axis_range);
