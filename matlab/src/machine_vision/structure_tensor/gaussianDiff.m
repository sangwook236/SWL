function [dzDx, dzDy] = gaussianDiff(z, sigma, dxs, dys)
%GAUSSIANDIFF Calculate derivatives using Gaussians.
%   [DZDX, DZDY] = GAUSSIANDIFF(Z, SIGMA, DX) computes the regularized
%   partial derivatives (DZDX and DZDY) of an image Z using a gaussian
%   with standard deviation SIGMA, assuming pixel spacing DX.

if nargin < 2
  sigma = 1;
end

if nargin < 3
  dxs = 1;
  dys = 1;
end

if nargin < 4
  dys = dxs;
end

%--S [] 2013/06/15: Sang-Wook Lee
%filtsize = 6*ceil(sigma) + 1;
filtsize = 2*ceil(sigma) + 1;
%--E [] 2013/06/15: Sang-Wook Lee
xs = linspace(-floor(filtsize/2), floor(filtsize/2), filtsize);

dgD = xs./(sqrt(2*pi)*sigma^3).*exp(-xs.^2./(2*sigma^2));
dzDx = filter2(dgD, z, 'same')./dxs;
dzDy = filter2(rot90(dgD), z, 'same')./dys;

% Z = fft2(z);
% Gx = fft2(fftshift(dgDx));
% Gy = fft2(fftshift(dgDy));
% 
% dzDx = real(ifft2(Z.*Gx))/dx;
% dzDy = real(ifft2(Z.*Gy))/dx;
