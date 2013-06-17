%--------------------------------------------------------------------

% [ref] https://bitbucket.org/dhr/odtgf

%addpath('D:\working_copy\swl_https\matlab\src\machine_vision\structure_tensor');

% CAUTION [] >> don't need to add two paths below
%addpath('D:\work_center\sw_dev\matlab\rnd\src\machine_vision\structure_tensor\dhr-odtgf-6f37207a76a5\Code\geom');
%addpath('D:\work_center\sw_dev\matlab\rnd\src\machine_vision\structure_tensor\dhr-odtgf-6f37207a76a5\Code\util');

%----------------------------------------------------------
% load images

image_file_list = [
	struct('filename', './hand/hand_01.jpg', 'rgb', true),
	struct('filename', './hand/hand_33.jpg', 'rgb', true),
	struct('filename', './hand/simple_hand_01.jpg', 'rgb', true),
	struct('filename', './hand/table_hand_01.jpg', 'rgb', true)
];
num_images = length(image_file_list);

original_input_images = cell(1, num_images);
for kk = 1:num_images
	% we must use double() instead of im2double().
	if image_file_list(kk).rgb
		original_input_images{kk} = double(imread(image_file_list(kk).filename));
	else
		original_input_images{kk} = double(imread(image_file_list(kk).filename));
	end;
	original_input_images{kk} = original_input_images{kk} / max(original_input_images{kk}(:));

	%imshow(processed_img);
end;

%--------------------------------------------------------------------
% structure tensor

sigma = 1;
rho = 1;
for kk = 1:num_images
	original_img = original_input_images{kk};
	%processed_img = rgb2gray(original_img);
	%processed_img = original_img(:,:,1);
	processed_img = rgb2hsv(original_img);
	processed_img = processed_img(:,:,1);

	[w1, w2, mu1, mu2] = structureTensor(processed_img, sigma, rho);

	% FIXME [check] >>
	w1 = real(w1);
	w2 = real(w2);
	mu1 = real(mu1);
	mu2 = real(mu2);

	%--------------------------------------------------------------------
	% post-processing

	% [ref] http://en.wikipedia.org/wiki/Structure_tensor

	[rows, cols] = size(processed_img);
	coherence = zeros(rows, cols);
	angle1 = zeros(rows, cols);
	angle2 = zeros(rows, cols);
	ev_ratio = zeros(rows, cols);
	for rr = 1:rows
		for cc = 1:cols
			if mu1(rr,cc) ~= 0 || mu2(rr,cc) ~= 0
				coherence(rr, cc) = ((mu1(rr,cc) - mu2(rr,cc)) / (mu1(rr,cc) + mu2(rr,cc)))^2;
				ev_ratio(rr, cc) = mu1(rr,cc) / mu2(rr,cc);
			end;
			if ~isnan(w1(rr,cc,1)) && ~isnan(w1(rr,cc,2))
				angle1(rr, cc) = atan2(w1(rr,cc,2), w1(rr,cc,1));
			end;
			if ~isnan(w2(rr,cc,1)) && ~isnan(w2(rr,cc,2))
				angle2(rr, cc) = atan2(w2(rr,cc,2), w2(rr,cc,1));
			end;
		end;
	end;

	%--------------------------------------------------------------------
	% show results

	figure;
	subplot(3,3,1), imshow(original_img);
	axis equal;
	subplot(3,3,2), imshow(processed_img);
	axis equal;
	subplot(3,3,4), imshow(coherence);
	axis equal;
	subplot(3,3,5), imshow(angle1);
	axis equal;
	subplot(3,3,6), imshow(angle2);
	axis equal;
	subplot(3,3,7), imshow(ev_ratio);
	axis equal;
	subplot(3,3,8), imshow(ev_ratio < 0.1);
	axis equal;
end;
