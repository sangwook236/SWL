%----------------------------------------------------------
%addpath('D:\work_center\sw_dev\matlab\rnd\src\machine_vision\camera_calibration_toolbox_for_matlab\toolbox_calib');

%----------------------------------------------------------

ir_image_file_list = [
	struct('filename', 'kinect_depth_20130530T103805.png', 'rgb', false),
	struct('filename', 'kinect_depth_20130531T023152.png', 'rgb', false), 
	struct('filename', 'kinect_depth_20130531T023346.png', 'rgb', false), 
	struct('filename', 'kinect_depth_20130531T023359.png', 'rgb', false) 
];
rgb_image_file_list = [
	struct('filename', 'kinect_rgba_20130530T103805.png', 'rgb', true),
	struct('filename', 'kinect_rgba_20130531T023152.png', 'rgb', true),
	struct('filename', 'kinect_rgba_20130531T023346.png', 'rgb', true),
	struct('filename', 'kinect_rgba_20130531T023359.png', 'rgb', true)
];
num_image_pairs = length(ir_image_file_list);

applied_method = 1;  % slow
applied_method = 2;  % fast
applied_method = 3;  % fastest

%----------------------------------------------------------

if true
	% IR (left) to RGB (right)
	% the 5th distortion parameter, kc(5) is activated.

	fc_ir = [ 5.865281297534211e+02     5.866623900166177e+02 ];  % [pixel]
	cc_ir = [ 3.371860463542209e+02     2.485298169373497e+02 ];  % [pixel]
	alpha_c_ir = [ 0.00000 ];
	kc_ir = [ -1.227084070414958e-01     5.027511830344261e-01    -2.562850607972214e-03     6.916249031489476e-03    -5.507709925923052e-01 ];

	fc_rgb = [ 5.248648751941851e+02     5.268281060449414e+02 ];  % [pixel]
	cc_rgb = [ 3.267484107269922e+02     2.618261807606497e+02 ];  % [pixel]
	alpha_c_rgb = [ 0.00000 ];
	kc_rgb = [ 2.796770514235670e-01    -1.112507253647945e+00     9.265501548915561e-04     2.428229310663184e-03     1.744019737212440e+00 ];

	R_ir_to_rgb = rodrigues([ -1.936270295074452e-03     1.331596538715070e-02     3.404073398703758e-03 ]);
	T_ir_to_rgb = [ 2.510788316141147e+01     4.096406849871768e+00    -5.759569165289306e+00 ]';  % [mm]

	KK_ir = [
		fc_ir(1) alpha_c_ir * fc_ir(1) cc_ir(1)
		0 fc_ir(2) cc_ir(2)
		0 0 1
	];

	KK_rgb = [
		fc_rgb(1) alpha_c_rgb * fc_rgb(1) cc_rgb(1)
		0 fc_rgb(2) cc_rgb(2)
		0 0 1
	];

	ncols_ir = 640;  % x
	nrows_ir = 480;  % y
	ncols_rgb = 640;  % x
	nrows_rgb = 480;  % y
elseif false
	% IR (left) to RGB (right)
	% the 5th distortion parameter, kc(5) is deactivated.

	fc_ir = [ 5.864902565580264e+02     5.867305900503998e+02 ];  % [pixel]
	cc_ir = [ 3.376088045224677e+02     2.480083390372575e+02 ];  % [pixel]
	alpha_c_ir = [ 0.00000 ];
	kc_ir = [ -1.123867977947529e-01     3.552017514491446e-01    -2.823972305243438e-03     7.246763414437084e-03    0.0 ];

	fc_rgb = [ 5.256215953836251e+02     5.278165866956751e+02 ];  % [pixel]
	cc_rgb = [ 3.260532981578608e+02     2.630788286947369e+02 ];  % [pixel]
	alpha_c_rgb = [ 0.00000 ];
	kc_rgb = [ 2.394862387380747e-01    -5.840355691714197e-01     2.567740590187774e-03     2.044179978023951e-03     0.0 ];

	R_ir_to_rgb = rodrigues([ 1.121432126402549e-03     1.535221550916760e-02     3.701648572107407e-03 ]);
	T_ir_to_rgb = [ 2.508484330268557e+01     3.773933285682256e+00    -4.725374631663055e+00 ]';  % [mm]

	KK_ir = [
		fc_ir(1) alpha_c_ir * fc_ir(1) cc_ir(1)
		0 fc_ir(2) cc_ir(2)
		0 0 1
	];

	KK_rgb = [
		fc_rgb(1) alpha_c_rgb * fc_rgb(1) cc_rgb(1)
		0 fc_rgb(2) cc_rgb(2)
		0 0 1
	];

	ncols_ir = 640;  % x
	nrows_ir = 480;  % y
	ncols_rgb = 640;  % x
	nrows_rgb = 480;  % y
elseif false
	% in case of using Calib_Results_stereo_rectified.mat.
	% these parameters are valid only for the rectified images.

	%load('Calib_Results_stereo_rectified.mat');
	%load('Calib_Results_stereo_rectified_wo_k5.mat');  % when k(5) = 0

	fc_ir = fc_left_new;
	cc_ir = cc_left_new;
	alpha_c_ir = alpha_c_left_new;
	kc_ir = kc_left_new;
	ncols_ir = nx_left_new;
	nrows_ir = ny_left_new;

	fc_rgb = fc_right_new;
	cc_rgb = cc_right_new;
	alpha_c_rgb = alpha_c_right_new;
	kc_rgb = kc_right_new;
	ncols_rgb = nx_right_new;
	nrows_rgb = ny_right_new;

	R_ir_to_rgb = R_new;
	T_ir_to_rgb = T_new;
else
	% in case of using Calib_Results_stereo.mat.

	%load('Calib_Results_stereo.mat');
	%load('Calib_Results_stereo_wo_k5.mat');  % when k(5) = 0

	fc_ir = fc_left;
	cc_ir = cc_left;
	alpha_c_ir = alpha_c_left;
	kc_ir = kc_left;
	ncols_ir = nx;
	nrows_ir = ny;

	fc_rgb = fc_right;
	cc_rgb = cc_right;
	alpha_c_rgb = alpha_c_right;
	kc_rgb = kc_right;
	ncols_rgb = nx;
	nrows_rgb = ny;

	R_ir_to_rgb = R;
	T_ir_to_rgb = T;
end;

if false
	% compute the new KK matrix to fit as much data in the image (in order to accomodate large distortions:
	r2_extreme_ir = (ncols_ir^2/(4*fc_ir(1)^2) + nrows_ir^2/(4*fc_ir(2)^2));
	dist_amount_ir = 1; %(1+kc_ir(1)*r2_extreme_ir + kc_ir(2)*r2_extreme_ir^2);
	fc_ir_new = dist_amount_ir * fc_ir;
	KK_ir_new = [fc_ir_new(1) alpha_c_ir*fc_ir_new(1) cc_ir(1); 0 fc_ir_new(2) cc_ir(2) ; 0 0 1];

	r2_extreme_rgb = (ncols_rgb^2/(4*fc_rgb(1)^2) + nrows_rgb^2/(4*fc_rgb(2)^2));
	dist_amount_rgb = 1; %(1+kc_rgb(1)*r2_extreme_rgb + kc_rgb(2)*r2_extreme_rgb^2);
	fc_rgb_new = dist_amount_rgb * fc_rgb;
	KK_rgb_new = [fc_rgb_new(1) alpha_c_rgb*fc_rgb_new(1) cc_rgb(1); 0 fc_rgb_new(2) cc_rgb(2) ; 0 0 1];

	fc_ir = fc_ir_new;
	KK_ir = KK_ir_new;
	fc_rgb = fc_rgb_new;
	KK_rgb = KK_rgb_new;
end;

%R_right_left = R_ir_to_rgb';
%T_right_left = -R_ir_to_rgb' * T_ir_to_rgb;
R_right_left = R_ir_to_rgb;
T_right_left = T_ir_to_rgb;

%----------------------------------------------------------

% [ref]
%	${CAMEARA_CALIBRATION_TOOLBOX_FOR_MATLAB_HOME}/undistort_image.m
%	${CAMEARA_CALIBRATION_TOOLBOX_FOR_MATLAB_HOME}/rect.m
%	${CAMEARA_CALIBRATION_TOOLBOX_FOR_MATLAB_HOME}/apply_distortion.m

%----------------------------------------------------------
% load images

ir_input_images = cell(1, num_image_pairs);
rgb_input_images = cell(1, num_image_pairs);
for kk = 1:num_image_pairs
	% we must use double() instead of im2double().
	if ir_image_file_list(kk).rgb
		ir_input_images{kk} = double(rgb2gray(imread(ir_image_file_list(kk).filename)));
	else
		ir_input_images{kk} = double(imread(ir_image_file_list(kk).filename));
	end;
	if rgb_image_file_list(kk).rgb
		rgb_input_images{kk} = double(rgb2gray(imread(rgb_image_file_list(kk).filename)));
	else
		rgb_input_images{kk} = double(imread(rgb_image_file_list(kk).filename));
	end;
end;

%----------------------------------------------------------
% undistort images

% TODO [check] >> is undistortion required before rectification?
%	I think undistortion process don't be required before rectification.
%	During rectification process, image undistortion is applied. (?)
if false
	for kk = 1:num_image_pairs
		msg = sprintf('undistorting %s & %s ...', ir_image_file_list(kk).filename, rgb_image_file_list(kk).filename);
		disp(msg);
		tic;

		ir_input_images{kk} = rect(ir_input_images{kk}, eye(3), fc_ir, cc_ir, kc_ir, KK_ir);
		rgb_input_images{kk} = rect(rgb_input_images{kk}, eye(3), fc_rgb, cc_rgb, kc_rgb, KK_rgb);

		toc;
	end;
end;

%----------------------------------------------------------
% rectify images

if 1 == applied_method
	for kk = 1:num_image_pairs
		Img_ir = ir_input_images{kk};
		Img_rgb = rgb_input_images{kk};

		% the left image is mapped onto the right image.
		Img_ir_mapped = zeros(size(Img_rgb));

		msg = sprintf('rectifying the %d-st image pair ...', kk);
		disp(msg);
		tic;

		KK_left_inv = inv(KK_ir);
		for yy = 1:nrows_ir
			for xx = 1:ncols_ir
				depth = Img_ir(yy, xx);

				% image coordinates (left) -> image coordinates (right)
				x_img_right = KK_rgb * (R_right_left * (depth * KK_left_inv * [ xx-1 ; yy-1 ; 1 ]) + T_right_left);
				x_img_right = round(x_img_right / x_img_right(3));

				if 0 <= x_img_right(1) && x_img_right(1) < ncols_rgb && 0 <= x_img_right(2) && x_img_right(2) < nrows_rgb
					Img_ir_mapped(x_img_right(2)+1, x_img_right(1)+1) = depth;
				end;
			end;
		end;

		toc;

		ir_output_images{kk} = Img_ir_mapped;
		rgb_output_images{kk} = Img_rgb;
	end;
elseif 2 == applied_method
	fx_d = fc_ir(1);
	fy_d = fc_ir(2);
	cx_d = cc_ir(1);
	cy_d = cc_ir(2);

	fx_rgb = fc_rgb(1);
	fy_rgb = fc_rgb(2);
	cx_rgb = cc_rgb(1);
	cy_rgb = cc_rgb(2);

	for kk = 1:num_image_pairs
		Img_ir = ir_input_images{kk};
		Img_rgb = rgb_input_images{kk};

		% the left image is mapped onto the right image.
		Img_ir_mapped = zeros(size(Img_rgb));

		msg = sprintf('rectifying the %d-st image pair ...', kk);
		disp(msg);
		tic;

		for yy = 1:nrows_ir
			for xx = 1:ncols_ir
				depth = Img_ir(yy, xx);

				% camera coordinates (left) = world coordinates
				X_camera_left = [ (xx-1 - cx_d) * depth / fx_d ; (yy-1 - cy_d) * depth / fy_d ; depth ];

				% camera coordinates (right)
				X_camera_right = R_right_left * X_camera_left + T_right_left;

				% image coordinates (right)
				x_img_right = round([ fx_rgb * X_camera_right(1) / X_camera_right(3) + cx_rgb ; fy_rgb * X_camera_right(2) / X_camera_right(3) + cy_rgb ]);

				%if 0 <= x_img_right(1) && x_img_right(1) < ncols_rgb && 0 <= x_img_right(2) && x_img_right(2) < nrows_rgb && (0 == Img_ir_mapped(x_img_right(2)+1, x_img_right(1)+1) || depth < Img_ir_mapped(x_img_right(2)+1, x_img_right(1)+1))
				if 0 <= x_img_right(1) && x_img_right(1) < ncols_rgb && 0 <= x_img_right(2) && x_img_right(2) < nrows_rgb
					Img_ir_mapped(x_img_right(2)+1, x_img_right(1)+1) = depth;
				end;
			end;
		end;

		toc;

		ir_output_images{kk} = Img_ir_mapped;
		rgb_output_images{kk} = Img_rgb;
	end;
elseif 3 == applied_method
	% homogeneous image coordinates (left): zero-based coordinates
	[CC, RR] = meshgrid(1:ncols_ir, 1:nrows_ir);
	IC_homo_left = [ reshape(CC, 1, ncols_ir*nrows_ir)-1 ; reshape(RR, 1, ncols_ir*nrows_ir)-1 ; ones(1, ncols_ir*nrows_ir) ];

	% homogeneous normalized camera coordinates (left)
	CC_norm_left = inv(KK_ir) * IC_homo_left;

	for kk = 1:num_image_pairs
		Img_ir = ir_input_images{kk};
		Img_rgb = rgb_input_images{kk};

		% the left image is mapped onto the right image.
		Img_ir_mapped = zeros(size(Img_rgb));

		msg = sprintf('rectifying the %d-st image pair ...', kk);
		disp(msg);
		tic;

		% camera coordinates (left)
		CC_left = repmat(reshape(Img_ir, 1, ncols_ir*nrows_ir), [3 1]) .* CC_norm_left;

		% camera coordinates (right)
		%CC_right = R_right_left' * (CC_left - repmat(T_right_left, [1, ncols_ir*nrows_ir]));
		CC_right = R_right_left * CC_left + repmat(T_right_left, [1, ncols_ir*nrows_ir]);

		% homogeneous normalized camera coordinates (right)
		CC_norm_right = CC_right ./ repmat(CC_right(3, :), [3 1]);

		% homogeneous image coordinates (right)
		IC_homo_right = round(KK_rgb * CC_norm_right);
		IC_homo_right(1:2,:) = IC_homo_right(1:2,:) + 1;  % one-based coordinates

		%IDX1 = find(IC_homo_right(1,:) > 0);
		%IDX2 = find(IC_homo_right(1,:) <= ncols_rgb);
		%IDX3 = find(IC_homo_right(2,:) > 0);
		%IDX4 = find(IC_homo_right(2,:) <= nrows_rgb);
		%IDX = intersect(intersect(IDX1, IDX2), intersect(IDX3, IDX4));
		IDX = find((IC_homo_right(1,:) > 0) & (IC_homo_right(1,:) <= ncols_rgb) & (IC_homo_right(2,:) > 0) & (IC_homo_right(2,:) <= nrows_rgb));
		for ii = 1:length(IDX)
			depth1 = Img_ir_mapped(IC_homo_right(2,IDX(ii)), IC_homo_right(1,IDX(ii)));
			depth2 = CC_left(3,IDX(ii));
			%if 0 == depth1 || depth1 > depth2
				Img_ir_mapped(IC_homo_right(2,IDX(ii)), IC_homo_right(1,IDX(ii))) = depth2;
			%end;
		end;

		toc;

		ir_output_images{kk} = Img_ir_mapped;
		rgb_output_images{kk} = Img_rgb;
	end;
else
	error('improper applied method index')'
end;

%----------------------------------------------------------
% show results

for kk = 1:num_image_pairs
	img1 = ir_output_images{kk};
	img1(find(isnan(img1))) = 0;
	img1 = img1 ./ max(max(img1));
	img2 = rgb_output_images{kk} ./ max(max(rgb_output_images{kk}));
	imgBlend1 = imfuse(img1, img2, 'blend');

	figure;
	subplot(2,2,1), imshow(img1);
	subplot(2,2,2), imshow(img2);
	subplot(2,2,3), imshow(imgBlend1);
end;
