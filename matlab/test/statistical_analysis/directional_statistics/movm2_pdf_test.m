%addpath('../../../src/statistical_analysis/directional_statistics');

% plot a mixture of bivariate von Mises distributions on the torus in the 3-dimensional space
% [ref] http://en.wikipedia.org/wiki/Bivariate_von_Mises_distribution

mu = [ 210*pi/180 pi/2 315*pi/180 ];  % [0 2*pi]
nu = [ pi/4 pi*3/4 pi/4 ];  % [0 2*pi]
kappa1 = [ 200 200 20 ];
kappa2 = [ 200 200 20 ];
kappa3 = [ 0 100 0 ];
alpha = [ 0.2 0.3 0.5 ];
ezsurf(@(phi, psi) movm2_pdf(phi, psi, mu, nu, kappa1, kappa2, kappa3, alpha), [ 0 2*pi 0 2*pi ])
