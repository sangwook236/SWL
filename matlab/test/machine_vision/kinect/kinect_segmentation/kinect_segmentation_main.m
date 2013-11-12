%----------------------------------------------------------
% load images

rgb_image_file_list = [
	struct('filename', 'kinect_rgba_20130528T163408.png', 'rgb', true),
	struct('filename', 'kinect_rgba_20130530T103805.png', 'rgb', true),
	struct('filename', 'kinect_rgba_20130531T023152.png', 'rgb', true),
	struct('filename', 'kinect_rgba_20130531T023346.png', 'rgb', true),
	struct('filename', 'kinect_rgba_20130531T023359.png', 'rgb', true)
];
depth_image_file_list = [
	struct('filename', 'kinect_depth_20130528T163408.png', 'rgb', false),
	struct('filename', 'kinect_depth_20130530T103805.png', 'rgb', false),
	struct('filename', 'kinect_depth_20130531T023152.png', 'rgb', false),
	struct('filename', 'kinect_depth_20130531T023346.png', 'rgb', false),
	struct('filename', 'kinect_depth_20130531T023359.png', 'rgb', false)
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
	if depth_image_file_list(kk).rgb
		depth_input_images{kk} = double(rgb2gray(imread(depth_image_file_list(kk).filename)));
	else
		depth_input_images{kk} = double(imread(depth_image_file_list(kk).filename));
	end;
end;

%----------------------------------------------------------
% depth histogram

% hist & ksdensity
%	[ref] D:\working_copy\research_https\matlab\human_motion_analysis\THoG\plot_HoG_and_MovM.m

figure;
for kk = 1:num_image_pairs
	%subplot(2, 3, kk), hist(depth_input_images{kk}(:), 1000);
	subplot(2, 3, kk), ksdensity(depth_input_images{kk}(:), 'kernel', 'normal', 'function', 'pdf');
end;

%----------------------------------------------------------
% show results

for ii = 1:num_image_pairs
	img1 = rgb_output_images{kk} ./ max(max(rgb_output_images{kk}));
	img2 = depth_output_images{kk};
	img2(find(isnan(img2))) = 0;
	img2 = img2 ./ max(max(img2));
	imgBlend1 = imfuse(img1, img2, 'blend');

	figure;
	subplot(2,2,1), imshow(img1);
	subplot(2,2,2), imshow(img2);
	subplot(2,2,3), imshow(imgBlend1);
end;
