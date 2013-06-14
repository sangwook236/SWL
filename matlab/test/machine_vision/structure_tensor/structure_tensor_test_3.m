%--------------------------------------------------------------------

% [ref] http://www.cs.cmu.edu/~sarsen/structureTensorTutorial/

%addpath('D:\work_center\sw_dev\matlab\rnd\src\machine_vision\structure_tensor\ST_pub');

%--------------------------------------------------------------------

img = double(imread('circle.pgm'));
%img = double(imread('oneColor.pgm'));
%img = double(imread('rampStep.pgm'));
%img = double(imread('stepEdge.pgm'));
%img = double(imread('circuit.tif'));
img = img / max(img(:));
%imshow(img);
%imagesc(img);

[rows, cols] = size(img);

maskSize = 3; 
DoG = difference_of_gaussian_kernels(maskSize);

Ix = conv2(img, DoG.Gx, 'same');
Iy = conv2(img, DoG.Gy, 'same');

%midpt = ceil(maskSize / 2);
%IxI = Ix(midpt, midpt);
%IyI = Iy(midpt, midpt); 

coherence = zeros(rows, cols);
for rr = 11:15
	for cc = 11:15
		ix = Ix(rr, cc);
		iy = Iy(rr, cc);

		% 2-dim structure tensor.
		ST = [
			ix*ix, ix*iy
		    ix*iy, iy*iy
		];

		[V D] = eig(ST)

		if D(1,1) ~= 0 || D(2,2) ~= 0
			coherence(rr, cc) = ((D(1,1) - D(2,2)) / (D(1,1) + D(2,2)))^2;
		end;
	end;
end;

figure;
subplot(1,2,1), imshow(img);
axis equal;
subplot(1,2,2), imshow(coherence);
axis equal;
