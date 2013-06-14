%--------------------------------------------------------------------

% [ref] https://bitbucket.org/dhr/odtgf

%addpath('D:\working_copy\swl_https\matlab\src\machine_vision\structure_tensor');

% CAUTION [] >> don't need to add two paths below
%addpath('D:\work_center\sw_dev\matlab\rnd\src\machine_vision\structure_tensor\dhr-odtgf-6f37207a76a5\Code\geom');
%addpath('D:\work_center\sw_dev\matlab\rnd\src\machine_vision\structure_tensor\dhr-odtgf-6f37207a76a5\Code\util');

%----------------------------------------------------------
% load images

rgb_image_file_list = [
	struct('filename', 'kinect_rgba_rectified_20130614T162309.png', 'rgb', true),
	struct('filename', 'kinect_rgba_rectified_20130614T162314.png', 'rgb', true),
	struct('filename', 'kinect_rgba_rectified_20130614T162348.png', 'rgb', true),
	struct('filename', 'kinect_rgba_rectified_20130614T162459.png', 'rgb', true),
	struct('filename', 'kinect_rgba_rectified_20130614T162525.png', 'rgb', true)
	struct('filename', 'kinect_rgba_rectified_20130614T162552.png', 'rgb', true)
];
depth_image_file_list = [
	struct('filename', 'kinect_depth_rectified_valid_20130614T162309.png', 'rgb', false),
	struct('filename', 'kinect_depth_rectified_valid_20130614T162314.png', 'rgb', false),
	struct('filename', 'kinect_depth_rectified_valid_20130614T162348.png', 'rgb', false),
	struct('filename', 'kinect_depth_rectified_valid_20130614T162459.png', 'rgb', false),
	struct('filename', 'kinect_depth_rectified_valid_20130614T162525.png', 'rgb', false)
	struct('filename', 'kinect_depth_rectified_valid_20130614T162552.png', 'rgb', false)
];
num_image_pairs = length(rgb_image_file_list);

rgb_input_images = cell(1, num_image_pairs);
depth_input_images = cell(1, num_image_pairs);
for kk = 1:num_image_pairs
	% we must use double() instead of im2double().
	if rgb_image_file_list(kk).rgb
		rgb_input_images{kk} = double(rgb2gray(imread(rgb_image_file_list(kk).filename)));
	else
		rgb_input_images{kk} = double(imread(rgb_image_file_list(kk).filename));
	end;
	rgb_input_images{kk} = rgb_input_images{kk} / max(rgb_input_images{kk}(:));

	if depth_image_file_list(kk).rgb
		depth_input_images{kk} = double(rgb2gray(imread(depth_image_file_list(kk).filename)));
	else
		depth_input_images{kk} = double(imread(depth_image_file_list(kk).filename));
	end;
	depth_input_images{kk} = depth_input_images{kk} / max(depth_input_images{kk}(:));

	%imshow(depth_img);
	%imagesc(depth_img);
end;

%--------------------------------------------------------------------
% structure tensor

sigma = 3;
rho = 2;
for kk = 1:num_image_pairs
	depth_img = depth_input_images{kk};
	rgb_img = rgb_input_images{kk};

	[w1, w2, mu1, mu2] = structureTensor(depth_img, sigma, rho);

	% FIXME [check] >>
	w1 = real(w1);
	w2 = real(w2);
	mu1 = real(mu1);
	mu2 = real(mu2);

	%--------------------------------------------------------------------
	% post-processing

	[rows, cols] = size(depth_img);
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
	% save results

	filename = sprintf('structure_tensor_ev_ratio_%d.png', kk);
	imwrite(ev_ratio, filename);

	%--------------------------------------------------------------------
	% show results

	figure;
	subplot(2,2,1), imshow(depth_img);
	axis equal;
	subplot(2,2,2), imshow(coherence);
	axis equal;
	subplot(2,2,3), imshow(angle1);
	axis equal;
	%subplot(2,2,4), imshow(angle2);
	subplot(2,2,4), imshow(ev_ratio);
	axis equal;

	%--------------------------------------------------------------------

	img1 = rgb_img;
	img2 = ev_ratio ./ max(max(ev_ratio));
	img3 = img2 > 0.05;
	imgBlend1 = imfuse(img1, img3, 'blend');

	figure;
	subplot(2,2,1), imshow(img1);
	subplot(2,2,2), imshow(img2);
	subplot(2,2,3), imshow(img3);
	subplot(2,2,4), imshow(imgBlend1);
end;
