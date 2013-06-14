function [w1, w2, mu1, mu2] = structureTensor(I, sigma, rho)
%
%   [w1, w2, mu1, mu2] = structureTensor(I, sigma, rho)
%
% At each location we compute the structure tensor:
%
% J (x,y) = grad(I) * grad(I)' = [Ix^2 Ix*Iy ; Iy*Ix Iy^2]
%
% The gradient is regularized by filtering with a gausian of std 'sigma'
%
% The structure tensor is regularized by filtering with a gaussian of std 'rho'
%
% The outputs w1 and w2 are the eigenvectors of the structure tensor at each location
% and mu1, mu2 are the associated eigenvalues (mu1 < mu2)

[nrows, ncols, nc] = size(I);

%RF% grad(I)
[dzDx, dzDy] = gaussianDiff(I, sigma);

%RF% Structure tensor:
Jxx = dzDx.^2;
Jxy = dzDx.*dzDy;
Jyy = dzDy.^2;

%RF% filtering the structure tensor:
%--S [] 2013/06/15: Sang-Wook Lee
%Jxx = gaussian(Jxx, rho);
%Jxy = gaussian(Jxy, rho);
%Jyy = gaussian(Jyy, rho);
Jxx = gaussianBlur(Jxx, rho);
Jxy = gaussianBlur(Jxy, rho);
Jyy = gaussianBlur(Jyy, rho);
%--E [] 2013/06/15: Sang-Wook Lee

w1 = zeros(nrows, ncols, 2);
w2 = zeros(nrows, ncols, 2);

%RF% compute eigenvectors and eigenvalues
a11 = Jxx; a12 = Jxy; a21 = Jxy; a22 = Jyy;
detJ = a11.*a22 - a12.*a21;

delta = sqrt((a11 + a22).^2 - 4*detJ);
mu1 = ((a11 + a22) + delta)/2;
mu2 = ((a11 + a22) - delta)/2;

j = find(abs(mu2) < abs(mu1));
tmp = mu2(j);
mu2(j) = mu1(j);
mu1(j) = tmp;

w1(:,:,1) = -a12;
w1(:,:,2) = (a11 - mu1);

w2(:,:,1) = -a12;
w2(:,:,2) = (a11 - mu2);

w1 = bsxfun(@rdivide, w1, sqrt(sum(w1.^2, 3)));
w2 = bsxfun(@rdivide, w2, sqrt(sum(w2.^2, 3)));