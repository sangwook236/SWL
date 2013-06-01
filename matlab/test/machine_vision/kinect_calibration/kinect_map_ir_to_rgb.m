%----------------------------------------------------------
%addpath('D:\work_center\sw_dev\matlab\rnd\src\machine_vision\camera_calibration_toolbox_for_matlab\toolbox_calib');
%addpath('D:\working_copy\swl_https\matlab\src\machine_vision');

%----------------------------------------------------------

left_images = [
	struct('filename', 'kinect_depth_20130530T103805.png', 'rgb', false),
	struct('filename', 'kinect_depth_20130531T023152.png', 'rgb', false), 
	struct('filename', 'kinect_depth_20130531T023346.png', 'rgb', false), 
	struct('filename', 'kinect_depth_20130531T023359.png', 'rgb', false) 
];
right_images = [
	struct('filename', 'kinect_rgba_20130530T103805.png', 'rgb', true),
	struct('filename', 'kinect_rgba_20130531T023152.png', 'rgb', true),
	struct('filename', 'kinect_rgba_20130531T023346.png', 'rgb', true),
	struct('filename', 'kinect_rgba_20130531T023359.png', 'rgb', true)
];
num_image_pairs = length(left_images);

applied_method = 1;  % slow
applied_method = 2;  % fast
applied_method = 3;  % fastest

%----------------------------------------------------------

if true
	% IR (left) to RGB (right)

	%Focal Length:          fc_left = [ 5.865259629841016e+02   5.869119946890888e+02 ] ? [ 6.84032   6.51357 ]
	%Principal point:       cc_left = [ 3.374030882445484e+02   2.486851671279394e+02 ] ? [ 12.06671   9.34551 ]
	%Skew:             alpha_c_left = [ 0.00000 ] ? [ 0.00000  ]   => angle of pixel axes = 90.00000 ? 0.00000 degrees
	%Distortion:            kc_left = [ -1.191900881116634e-01   4.770987342499287e-01   -2.359540729860922e-03   6.980497982277759e-03   -4.910177215685992e-01 ] ? [ 0.12402   1.33846   0.00518   0.00666  4.24087 ]

	%Focal Length:          fc_right = [ 5.246059127053605e+02   5.268025156310126e+02 ] ? [ 6.25420   5.97153 ]
	%Principal point:       cc_right = [ 3.278147550011708e+02   2.616335603485837e+02 ] ? [ 9.47633   8.65563 ]
	%Skew:             alpha_c_right = [ 0.00000 ] ? [ 0.00000  ]   => angle of pixel axes = 90.00000 ? 0.00000 degrees
	%Distortion:            kc_right = [ 2.877942533299389e-01  -1.183365423881459e+00   8.819616313287803e-04   3.334954093546241e-03   1.924070186169354e+00 ] ? [ 0.14257   1.58175   0.00820   0.00857  5.20374 ]

	%Rotation vector:             om = [ -2.563731899937744e-03   1.170121176120723e-02   3.398860594633926e-03 ] ? [ 0.01441   0.02317  0.00110 ]
	%Translation vector:           T = [ 2.509097936007354e+01   4.114956573893061e+00   -6.430074873604823e+00 ] ? [ 1.26092   1.25532  6.49162 ]

	fc_left = [ 5.865259629841016e+02   5.869119946890888e+02 ];
	cc_left = [ 3.374030882445484e+02   2.486851671279394e+02 ];
	alpha_c_left = [ 0.00000 ];
	kc_left = [ -1.191900881116634e-01   4.770987342499287e-01   -2.359540729860922e-03   6.980497982277759e-03   -4.910177215685992e-01 ];

	fc_right = [ 5.246059127053605e+02   5.268025156310126e+02 ];
	cc_right = [ 3.278147550011708e+02   2.616335603485837e+02 ];
	alpha_c_right = [ 0.00000 ];
	kc_right = [ 2.877942533299389e-01  -1.183365423881459e+00   8.819616313287803e-04   3.334954093546241e-03   1.924070186169354e+00 ];

	R = rodrigues([ -2.563731899937744e-03   1.170121176120723e-02   3.398860594633926e-03 ]);
	T = [ 2.509097936007354e+01   4.114956573893061e+00   -6.430074873604823e+00 ]';

	KK_left = [
		fc_left(1) alpha_c_left * fc_left(1) cc_left(1)
		0 fc_left(2) cc_left(2)
		0 0 1
	];

	KK_right = [
		fc_right(1) alpha_c_right * fc_right(1) cc_right(1)
		0 fc_right(2) cc_right(2)
		0 0 1
	];

	ncols_left = 640;  % x
	nrows_left = 480;  % y
	ncols_right = 640;  % x
	nrows_right = 480;  % y
elseif false
	% in case of using Calib_Results_stereo_rectified.mat.
	% these parameters are valid only for the rectified images.

	fc_left = fc_left_new;
	cc_left = cc_left_new;
	alpha_c_left = alpha_c_left_new;
	kc_left = kc_left_new;
	ncols_left = nx_left_new;
	nrows_left = ny_left_new;

	fc_right = fc_right_new;
	cc_right = cc_right_new;
	alpha_c_right = alpha_c_right_new;
	kc_right = kc_right_new;
	ncols_right = nx_right_new;
	nrows_right = ny_right_new;

	R = R_new;
	T = T_new;
else
	% in case of using Calib_Results_stereo.mat.
	%	e.g.) load('Calib_Results_stereo.mat');

	%fc_left = fc_left;
	%cc_left = cc_left;
	%alpha_c_left = alpha_c_left;
	%kc_left = kc_left;
	%ncols_left = nx;
	%nrows_left = ny;

	%fc_right = fc_right;
	%cc_right = cc_right;
	%alpha_c_right = alpha_c_right;
	%kc_right = kc_right;
	%ncols_right = nx;
	%nrows_right = ny;

	%R = R;
	%T = T;
end;

if false
	% compute the new KK matrix to fit as much data in the image (in order to accomodate large distortions:
	r2_extreme_left = (ncols_left^2/(4*fc_left(1)^2) + nrows_left^2/(4*fc_left(2)^2));
	dist_amount_left = 1; %(1+kc_left(1)*r2_extreme_left + kc_left(2)*r2_extreme_left^2);
	fc_left_new = dist_amount_left * fc_left;
	KK_left_new = [fc_left_new(1) alpha_c_left*fc_left_new(1) cc_left(1); 0 fc_left_new(2) cc_left(2) ; 0 0 1];

	r2_extreme_right = (ncols_right^2/(4*fc_right(1)^2) + nrows_right^2/(4*fc_right(2)^2));
	dist_amount_right = 1; %(1+kc_right(1)*r2_extreme_right + kc_right(2)*r2_extreme_right^2);
	fc_right_new = dist_amount_right * fc_right;
	KK_right_new = [fc_right_new(1) alpha_c_right*fc_right_new(1) cc_right(1); 0 fc_right_new(2) cc_right(2) ; 0 0 1];

	fc_left = fc_left_new;
	KK_left = KK_left_new;
	fc_right = fc_right_new;
	KK_right = KK_right_new;
end;

%----------------------------------------------------------

% [ref]
%	${CAMEARA_CALIBRATION_TOOLBOX_FOR_MATLAB_HOME}/undistort_image.m
%	${CAMEARA_CALIBRATION_TOOLBOX_FOR_MATLAB_HOME}/rect.m
%	${CAMEARA_CALIBRATION_TOOLBOX_FOR_MATLAB_HOME}/apply_distortion.m

%----------------------------------------------------------

%R_right_left = R';
%T_right_left = -R' * T;
R_right_left = R;
T_right_left = T;

input_images_left = cell(1, num_image_pairs);
input_images_right = cell(1, num_image_pairs);
for kk = 1:num_image_pairs
	% load images
	% we must use double() instead of im2double().
	if left_images(kk).rgb
		Img_left = double(rgb2gray(imread(left_images(kk).filename)));
	else
		Img_left = double(imread(left_images(kk).filename));
	end;
	if right_images(kk).rgb
		Img_right = double(rgb2gray(imread(right_images(kk).filename)));
	else
		Img_right = double(imread(right_images(kk).filename));
	end;
	Img_left_mapped = zeros(size(Img_right));  % depth image (left image) is mapped onto RGB image (right image).

	% undistort images
	sprintf('undistorting %s & %s ...', left_images(kk).filename, right_images(kk).filename)
	tic;

	Img_left = rect(Img_left, eye(3), fc_left, cc_left, kc_left, KK_left);
	Img_right = rect(Img_right, eye(3), fc_right, cc_right, kc_right, KK_right);

	toc;

	input_images_left{kk} = Img_left;
	input_images_right{kk} = Img_right;
end;

%----------------------------------------------------------
% rectify images

if 1 == applied_method
	for kk = 1:num_image_pairs
		Img_left = input_images_left{kk};
		Img_right = input_images_right{kk};

		% the left image is mapped onto the right image.
		Img_left_mapped = zeros(size(Img_right));

		sprintf('rectifying the %d-st image pair ...', kk)
		tic;

		KK_left_inv = inv(KK_left);
		for yy = 1:nrows_left
			for xx = 1:ncols_left
				depth = Img_left(yy, xx);

				% image coordinates (left) -> image coordinates (right)
				x_img_right = KK_right * (R_right_left * (depth * KK_left_inv * [ xx-1 ; yy-1 ; 1 ]) + T_right_left);
				x_img_right = round(x_img_right / x_img_right(3));

				if 0 <= x_img_right(1) && x_img_right(1) < ncols_right && 0 <= x_img_right(2) && x_img_right(2) < nrows_right
					Img_left_mapped(x_img_right(2)+1, x_img_right(1)+1) = depth;
				end;
			end;
		end;

		toc;

		output_images_left{kk} = Img_left_mapped;
		output_images_right{kk} = Img_right;
	end;
elseif 2 == applied_method
	fx_d = fc_left(1);
	fy_d = fc_left(2);
	cx_d = cc_left(1);
	cy_d = cc_left(2);

	fx_rgb = fc_right(1);
	fy_rgb = fc_right(2);
	cx_rgb = cc_right(1);
	cy_rgb = cc_right(2);

	for kk = 1:num_image_pairs
		Img_left = input_images_left{kk};
		Img_right = input_images_right{kk};

		% the left image is mapped onto the right image.
		Img_left_mapped = zeros(size(Img_right));

		sprintf('rectifying the %d-st image pair ...', kk)
		tic;

		for yy = 1:nrows_left
			for xx = 1:ncols_left
				depth = Img_left(yy, xx);

				% camera coordinates (left) = world coordinates
				X_camera_left = [ (xx-1 - cx_d) * depth / fx_d ; (yy-1 - cy_d) * depth / fy_d ; depth ];

				% camera coordinates (right)
				X_camera_right = R_right_left * X_camera_left + T_right_left;

				% image coordinates (right)
				x_img_right = round([ fx_rgb * X_camera_right(1) / X_camera_right(3) + cx_rgb ; fy_rgb * X_camera_right(2) / X_camera_right(3) + cy_rgb ]);

				%if 0 <= x_img_right(1) && x_img_right(1) < ncols_right && 0 <= x_img_right(2) && x_img_right(2) < nrows_right && (0 == Img_left_mapped(x_img_right(2)+1, x_img_right(1)+1) || depth < Img_left_mapped(x_img_right(2)+1, x_img_right(1)+1))
				if 0 <= x_img_right(1) && x_img_right(1) < ncols_right && 0 <= x_img_right(2) && x_img_right(2) < nrows_right
					Img_left_mapped(x_img_right(2)+1, x_img_right(1)+1) = depth;
				end;
			end;
		end;

		toc;

		output_images_left{kk} = Img_left_mapped;
		output_images_right{kk} = Img_right;
	end;
elseif 3 == applied_method
	% homogeneous image coordinates (left): zero-based coordinates
	[CC, RR] = meshgrid(1:ncols_left, 1:nrows_left);
	IC_homo_left = [ reshape(CC, 1, ncols_left*nrows_left)-1 ; reshape(RR, 1, ncols_left*nrows_left)-1 ; ones(1, ncols_left*nrows_left) ];

	% homogeneous normalized camera coordinates (left)
	CC_norm_left = inv(KK_left) * IC_homo_left;

	for kk = 1:num_image_pairs
		Img_left = input_images_left{kk};
		Img_right = input_images_right{kk};

		% the left image is mapped onto the right image.
		Img_left_mapped = zeros(size(Img_right));

		sprintf('rectifying the %d-st image pair ...', kk)
		tic;

		% camera coordinates (left)
		CC_left = repmat(reshape(Img_left, 1, ncols_left*nrows_left), [3 1]) .* CC_norm_left;

		% camera coordinates (right)
		%CC_right = R_right_left' * (CC_left - repmat(T_right_left, [1, ncols_left*nrows_left]));
		CC_right = R_right_left * CC_left + repmat(T_right_left, [1, ncols_left*nrows_left]);

		% homogeneous normalized camera coordinates (right)
		CC_norm_right = CC_right ./ repmat(CC_right(3, :), [3 1]);

		% homogeneous image coordinates (right)
		IC_homo_right = round(KK_right * CC_norm_right);
		IC_homo_right(1:2,:) = IC_homo_right(1:2,:) + 1;  % one-based coordinates

		IDX1 = find(IC_homo_right(1,:) > 0);
		IDX2 = find(IC_homo_right(1,:) <= ncols_right);
		IDX3 = find(IC_homo_right(2,:) > 0);
		IDX4 = find(IC_homo_right(2,:) <= nrows_right);
		IDX = intersect(intersect(IDX1, IDX2), intersect(IDX3, IDX4));
		for ii = 1:length(IDX)
			depth1 = Img_left_mapped(IC_homo_right(2,IDX(ii)), IC_homo_right(1,IDX(ii)));
			depth2 = CC_left(3,IDX(ii));
			%if 0 == depth1 || depth1 > depth2
				Img_left_mapped(IC_homo_right(2,IDX(ii)), IC_homo_right(1,IDX(ii))) = depth2;
			%end;
		end;

		toc;

		output_images_left{kk} = Img_left_mapped;
		output_images_right{kk} = Img_right;
	end;
else
	error('improper applied method index')'
end;

%----------------------------------------------------------
% show results

for kk = 1:num_image_pairs
	img1 = output_images_left{kk};
	img1(find(isnan(img1))) = 0;
	img1 = img1 ./ max(max(img1));
	img2 = output_images_right{kk} ./ max(max(output_images_right{kk}));
	imgBlend1 = imfuse(img1, img2, 'blend');

	figure;
	subplot(2,2,1), imshow(img1);
	subplot(2,2,2), imshow(img2);
	subplot(2,2,3), imshow(imgBlend1);
end;
