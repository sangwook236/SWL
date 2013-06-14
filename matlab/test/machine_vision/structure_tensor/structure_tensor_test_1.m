%--------------------------------------------------------------------

% [ref] https://bitbucket.org/dhr/odtgf

%addpath('D:\working_copy\swl_https\matlab\src\machine_vision\structure_tensor');

% CAUTION [] >> don't need to add two paths below
%addpath('D:\work_center\sw_dev\matlab\rnd\src\machine_vision\structure_tensor\dhr-odtgf-6f37207a76a5\Code\geom');
%addpath('D:\work_center\sw_dev\matlab\rnd\src\machine_vision\structure_tensor\dhr-odtgf-6f37207a76a5\Code\util');

%--------------------------------------------------------------------

img = double(imread('circle.pgm'));
%img = double(imread('oneColor.pgm'));
%img = double(imread('rampStep.pgm'));
%img = double(imread('stepEdge.pgm'));
%img = double(imread('circuit.tif'));
img = img / max(img(:));
%imshow(img);
%imagesc(img);

sigma = 1;
rho = 1;
[w1, w2, mu1, mu2] = structureTensor(img, sigma, rho);

[rows, cols] = size(img);
coherence = zeros(rows, cols);
for rr = 1:rows
	for cc = 1:cols
		if mu1(rr,cc) ~= 0 || mu2(rr,cc) ~= 0
			coherence(rr, cc) = ((mu1(rr,cc) - mu2(rr,cc)) / (mu1(rr,cc) + mu2(rr,cc)))^2;
		end;
	end;
end;

figure;
subplot(1,2,1), imshow(img);
axis equal;
subplot(1,2,2), imshow(coherence);
axis equal;
