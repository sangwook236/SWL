%addpath('../../../src/statistical_analysis/directional_statistics');

% plot a mixture of von Mises distributions
mean_dir = [ 2*pi/18 2*pi/4 2*pi*2/3 ];
kappa = [ 1 2 1.5 ];
alpha = [ 0.2 0.3 0.5 ];
ezplot(@(x) movm_pdf(x, mean_dir, kappa, alpha), [ 0 2*pi ])
