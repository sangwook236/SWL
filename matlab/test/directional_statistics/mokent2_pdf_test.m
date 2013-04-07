%addpath('..\..\src\directional_statistics');

% plot a mixture of Kent distributions on the 2-dimensional sphere in the 3-dimensional space
kappa = [ 1000 10 200 ];
beta = [ 499 0 50 ];
gamma1 = [];
gamma2 = [];
gamma3 = [];
alpha = [ 0.2 0.3 0.5 ];
ezsurf(@(theta, phi) mokent2_pdf(theta, phi, kappa, beta, gamma1, gamma2, gamma3, alpha), [ 0 pi 0 2*pi ])
