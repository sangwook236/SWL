%addpath('../../../src/statistics/directional_statistics');

% plot a mixture of von Mises-Fisher distributions on the 2-dimensional sphere in the 3-dimensional space

numComponents = 3;

mu_theta = [ pi/2 pi/2 pi/4 ];
mu_phi = [ 0 0 330*pi/180 ];
mu = zeros(3, numComponents);
for ii = 1:numComponents
    mu(:,ii) = [ sin(mu_theta(ii))*cos(mu_phi(ii)) ; sin(mu_theta(ii))*sin(mu_phi(ii)) ; cos(mu_theta(ii)) ];
end;
kappa = [ 200 200 20 ];
alpha = [ 0.2 0.3 0.5 ];

ezsurf(@(theta, phi) movmf2_pdf(theta, phi, mu, kappa, alpha), [ 0 pi 0 2*pi ])
