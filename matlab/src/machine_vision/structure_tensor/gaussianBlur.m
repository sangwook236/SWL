function img = gaussianBlur(img, sigma)
%
% sigma = width of the gaussian.
%

%[nrow, ncols] = size(img);
%[x,y] = meshgrid(-4*sigma:1:4*sigma);
%r2 = x.^2 +y.^2;

%g = 1/(sqrt(2*pi)*sigma) * exp (-r2 / sigma^2);
%g = g /sum(g(:));

%--S [] 2013/06/15: Sang-Wook Lee
%gx = exp (- (-3*sigma:1:3*sigma).^2 / sigma^2);
gx = exp (- (-sigma:1:sigma).^2 / sigma^2);
%--E [] 2013/06/15: Sang-Wook Lee
gx = gx / sum(gx(:));
%size(gx)

img = convn(img, gx, 'same');
img = convn(img, gx', 'same');


