%----------------------------------------------------------
%addpath('D:\work_center\sw_dev\matlab\rnd\src\machine_vision\camera_calibration_toolbox_for_matlab\toolbox_calib');

%----------------------------------------------------------
% load images

rgb_image_file_list = [
	struct('filename', 'kinect_rgba_20130530T103805.png', 'rgb', true),
	struct('filename', 'kinect_rgba_20130531T023152.png', 'rgb', true),
	struct('filename', 'kinect_rgba_20130531T023346.png', 'rgb', true),
	struct('filename', 'kinect_rgba_20130531T023359.png', 'rgb', true)
];
ir_image_file_list = [
	struct('filename', 'kinect_depth_20130530T103805.png', 'rgb', false),
	struct('filename', 'kinect_depth_20130531T023152.png', 'rgb', false), 
	struct('filename', 'kinect_depth_20130531T023346.png', 'rgb', false), 
	struct('filename', 'kinect_depth_20130531T023359.png', 'rgb', false) 
];
num_image_pairs = length(rgb_image_file_list);

rgb_input_images = cell(1, num_image_pairs);
ir_input_images = cell(1, num_image_pairs);
for kk = 1:num_image_pairs
	% we must use double() instead of im2double().
	if rgb_image_file_list(kk).rgb
		rgb_input_images{kk} = double(rgb2gray(imread(rgb_image_file_list(kk).filename)));
	else
		rgb_input_images{kk} = double(imread(rgb_image_file_list(kk).filename));
	end;
	if ir_image_file_list(kk).rgb
		ir_input_images{kk} = double(rgb2gray(imread(ir_image_file_list(kk).filename)));
	else
		ir_input_images{kk} = double(imread(ir_image_file_list(kk).filename));
	end;
end;

%----------------------------------------------------------
% set parameters

applied_method = 1;  % slow
applied_method = 2;  % fast
applied_method = 3;  % fastest

%----------------------------------------------------------

if true
	% RGB (left) to IR (right)
	% the 5th distortion parameter, kc(5) is activated.

	fc_rgb = [ 5.248648079874888e+02     5.268280486062615e+02 ];  % [pixel]
	cc_rgb = [ 3.267487100838014e+02     2.618261169946102e+02 ];  % [pixel]
	alpha_c_rgb = [ 0.00000 ];
	kc_rgb = [ 2.796764337988712e-01    -1.112497355183840e+00     9.264749543097661e-04     2.428507887293728e-03     1.743975665436613e+00 ];

	fc_ir = [ 5.865282023957649e+02     5.866624209441105e+02 ];  % [pixel]
	cc_ir = [ 3.371875014947813e+02     2.485295493095561e+02 ];  % [pixel]
	alpha_c_ir = [ 0.00000 ];
	kc_ir = [ -1.227176734054719e-01     5.028746725848668e-01    -2.563029340202278e-03     6.916996280663117e-03    -5.512162545452755e-01 ];

	R_rgb_to_ir = rodrigues([ 1.935939237060295e-03    -1.331788958930441e-02    -3.404128236480992e-03 ]);
	T_rgb_to_ir = [ -2.519613785802952e+01    -4.021596559891519e+00     5.416691156831289e+00 ]';  % [mm]

	KK_rgb = [
		fc_rgb(1) alpha_c_rgb * fc_rgb(1) cc_rgb(1)
		0 fc_rgb(2) cc_rgb(2)
		0 0 1
	];

	KK_ir = [
		fc_ir(1) alpha_c_ir * fc_ir(1) cc_ir(1)
		0 fc_ir(2) cc_ir(2)
		0 0 1
	];

	ncols_rgb = 640;  % x
	nrows_rgb = 480;  % y
	ncols_ir = 640;  % x
	nrows_ir = 480;  % y
elseif false
	% RGB (left) to IR (right)
	% the 5th distortion parameter, kc(5) is deactivated.

	fc_rgb = [ 5.256217798767822e+02     5.278167798992870e+02 ];  % [pixel]
	cc_rgb = [ 3.260534767468189e+02     2.630800669346188e+02 ];  % [pixel]
	alpha_c_rgb = [ 0.00000 ];
	kc_rgb = [ 2.394861400525463e-01    -5.840298777969020e-01     2.568959896208732e-03     2.044336479083819e-03    0.0 ];

	fc_ir = [ 5.864904832545356e+02     5.867308191567271e+02 ];  % [pixel]
	cc_ir = [ 3.376079004969836e+02     2.480098376453992e+02 ];  % [pixel]
	alpha_c_ir = [ 0.00000 ];
	kc_ir = [ -1.123902857373373e-01     3.552211727724343e-01    -2.823183218548772e-03     7.246270574438420e-03   0.0 ];

	R_rgb_to_ir = rodrigues([ -1.121214964017936e-03    -1.535031632771925e-02    -3.701579055761772e-03 ]);
	T_rgb_to_ir = [ -2.516823022792618e+01    -3.675852559848838e+00     4.343837142667145e+00 ]';  % [mm]

	KK_rgb = [
		fc_rgb(1) alpha_c_rgb * fc_rgb(1) cc_rgb(1)
		0 fc_rgb(2) cc_rgb(2)
		0 0 1
	];

	KK_ir = [
		fc_ir(1) alpha_c_ir * fc_ir(1) cc_ir(1)
		0 fc_ir(2) cc_ir(2)
		0 0 1
	];

	ncols_rgb = 640;  % x
	nrows_rgb = 480;  % y
	ncols_ir = 640;  % x
	nrows_ir = 480;  % y
elseif false
	% in case of using Calib_Results_stereo_rectified.mat.
	% these parameters are valid only for the rectified images.

	%load('Calib_Results_stereo_rectified.mat');
	%load('Calib_Results_stereo_rectified_wo_k5.mat');  % when k(5) = 0

	fc_rgb = fc_left_new;
	cc_rgb = cc_left_new;
	alpha_c_rgb = alpha_c_left_new;
	kc_rgb = kc_left_new;
	ncols_rgb = nx_left_new;
	nrows_rgb = ny_left_new;

	fc_ir = fc_right_new;
	cc_ir = cc_right_new;
	alpha_c_ir = alpha_c_right_new;
	kc_ir = kc_right_new;
	ncols_ir = nx_right_new;
	nrows_ir = ny_right_new;

	R_rgb_to_ir = R_new;
	T_rgb_to_ir = T_new;
else
	% in case of using Calib_Results_stereo.mat.

	%load('Calib_Results_stereo.mat');
	%load('Calib_Results_stereo_wo_k5.mat');  % when k(5) = 0

	fc_rgb = fc_left;
	cc_rgb = cc_left;
	alpha_c_rgb = alpha_c_left;
	kc_rgb = kc_left;
	ncols_rgb = nx;
	nrows_rgb = ny;

	fc_ir = fc_right;
	cc_ir = cc_right;
	alpha_c_ir = alpha_c_right;
	kc_ir = kc_right;
	ncols_ir = nx;
	nrows_ir = ny;

	R_rgb_to_ir = R;
	T_rgb_to_ir = T;
end;

if false
	% compute the new KK matrix to fit as much data in the image (in order to accomodate large distortions:
	r2_extreme_rgb = (ncols_rgb^2/(4*fc_rgb(1)^2) + nrows_rgb^2/(4*fc_rgb(2)^2));
	dist_amount_rgb = 1; %(1+kc_rgb(1)*r2_extreme_rgb + kc_rgb(2)*r2_extreme_rgb^2);
	fc_rgb_new = dist_amount_rgb * fc_rgb;
	KK_rgb_new = [fc_rgb_new(1) alpha_c_rgb*fc_rgb_new(1) cc_rgb(1); 0 fc_rgb_new(2) cc_rgb(2) ; 0 0 1];

	r2_extreme_ir = (ncols_ir^2/(4*fc_ir(1)^2) + nrows_ir^2/(4*fc_ir(2)^2));
	dist_amount_ir = 1; %(1+kc_ir(1)*r2_extreme_ir + kc_ir(2)*r2_extreme_ir^2);
	fc_ir_new = dist_amount_ir * fc_ir;
	KK_ir_new = [fc_ir_new(1) alpha_c_ir*fc_ir_new(1) cc_ir(1); 0 fc_ir_new(2) cc_ir(2) ; 0 0 1];

	fc_rgb = fc_rgb_new;
	KK_rgb = KK_rgb_new;
	fc_ir = fc_ir_new;
	KK_ir = KK_ir_new;
end;

R_left_right = R_rgb_to_ir';
T_left_right = -R_rgb_to_ir' * T_rgb_to_ir;
%R_left_right = R_rgb_to_ir;
%T_left_right = T_rgb_to_ir;

%----------------------------------------------------------

% [ref]
%	${CAMEARA_CALIBRATION_TOOLBOX_FOR_MATLAB_HOME}/undistort_image.m
%	${CAMEARA_CALIBRATION_TOOLBOX_FOR_MATLAB_HOME}/rect.m
%	${CAMEARA_CALIBRATION_TOOLBOX_FOR_MATLAB_HOME}/apply_distortion.m

%----------------------------------------------------------
% undistort images

% TODO [check] >> is undistortion required before rectification?
%	I think undistortion process don't be required before rectification.
%	During rectification process, image undistortion is applied. (?)
if false
	for kk = 1:num_image_pairs
		msg = sprintf('undistorting %s & %s ...', rgb_image_file_list(kk).filename, ir_image_file_list(kk).filename);
		disp(msg);
		tic;

		rgb_input_images{kk} = rect(rgb_input_images{kk}, eye(3), fc_rgb, cc_rgb, kc_rgb, KK_rgb);
		ir_input_images{kk} = rect(ir_input_images{kk}, eye(3), fc_ir, cc_ir, kc_ir, KK_ir);

		toc;
	end;
end;

%----------------------------------------------------------
% rectify images

if 1 == applied_method
	for kk = 1:num_image_pairs
		Img_rgb = rgb_input_images{kk};
		Img_ir = ir_input_images{kk};

		% the right image is mapped onto the left image.
		Img_ir_mapped = zeros(size(Img_rgb));

		msg = sprintf('rectifying the %d-st image pair ...', kk);
		disp(msg);
		tic;

		KK_right_inv = inv(KK_ir);
		for yy = 1:nrows_ir
			for xx = 1:ncols_ir
				depth = Img_ir(yy, xx);

				% image coordinates (right) -> image coordinates (left)
				x_img_left = KK_rgb * (R_left_right * (depth * KK_right_inv * [ xx-1 ; yy-1 ; 1 ]) + T_left_right);
				x_img_left = round(x_img_left / x_img_left(3));

				if 0 <= x_img_left(1) && x_img_left(1) < ncols_rgb && 0 <= x_img_left(2) && x_img_left(2) < nrows_rgb
					Img_ir_mapped(x_img_left(2)+1, x_img_left(1)+1) = depth;
				end;
			end;
		end;

		toc;

		rgb_output_images{kk} = Img_rgb;
		ir_output_images{kk} = Img_ir_mapped;
	end;
elseif 2 == applied_method
	fx_rgb = fc_rgb(1);
	fy_rgb = fc_rgb(2);
	cx_rgb = cc_rgb(1);
	cy_rgb = cc_rgb(2);

	fx_d = fc_ir(1);
	fy_d = fc_ir(2);
	cx_d = cc_ir(1);
	cy_d = cc_ir(2);

	for kk = 1:num_image_pairs
		Img_rgb = rgb_input_images{kk};
		Img_ir = ir_input_images{kk};

		% the right image is mapped onto the left image.
		Img_ir_mapped = zeros(size(Img_rgb));

		msg = sprintf('rectifying the %d-st image pair ...', kk);
		disp(msg);
		tic;

		for yy = 1:nrows_ir
			for xx = 1:ncols_ir
				depth = Img_ir(yy, xx);

				% camera coordinates (right)
				X_camera_right = [ (xx-1 - cx_d) * depth / fx_d ; (yy-1 - cy_d) * depth / fy_d ; depth ];

				% camera coordinates (left) = world coordinates
				X_camera_left = R_left_right * X_camera_right + T_left_right;

				% image coordinates (left)
				x_img_left = round([ fx_rgb * X_camera_left(1) / X_camera_left(3) + cx_rgb ; fy_rgb * X_camera_left(2) / X_camera_left(3) + cy_rgb ]);

				%if 0 <= x_img_left(1) && x_img_left(1) < ncols_rgb && 0 <= x_img_left(2) && x_img_left(2) < nrows_rgb && (0 == Img_ir_mapped(x_img_left(2)+1, x_img_left(1)+1) || depth < Img_ir_mapped(x_img_left(2)+1, x_img_left(1)+1))
				if 0 <= x_img_left(1) && x_img_left(1) < ncols_rgb && 0 <= x_img_left(2) && x_img_left(2) < nrows_rgb
					Img_ir_mapped(x_img_left(2)+1, x_img_left(1)+1) = depth;
				end;
			end;
		end;

		toc;

		rgb_output_images{kk} = Img_rgb;
		ir_output_images{kk} = Img_ir_mapped;
	end;
elseif 3 == applied_method
	% homogeneous image coordinates (right): zero-based coordinates
	[CC, RR] = meshgrid(1:ncols_ir, 1:nrows_ir);
	IC_homo_right = [ reshape(CC, 1, ncols_ir*nrows_ir)-1 ; reshape(RR, 1, ncols_ir*nrows_ir)-1 ; ones(1, ncols_ir*nrows_ir) ];

	% homogeneous normalized camera coordinates (right)
	CC_norm_right = inv(KK_ir) * IC_homo_right;

	for kk = 1:num_image_pairs
		Img_rgb = rgb_input_images{kk};
		Img_ir = ir_input_images{kk};

		% the right image is mapped onto the left image.
		Img_ir_mapped = zeros(size(Img_rgb));

		msg = sprintf('rectifying the %d-st image pair ...', kk);
		disp(msg);
		tic;

		% camera coordinates (right)
		CC_right = repmat(reshape(Img_ir, 1, ncols_ir*nrows_ir), [3 1]) .* CC_norm_right;

		% camera coordinates (left)
		%CC_left = R_left_right' * (CC_right - repmat(T_left_right, [1, ncols_ir*nrows_ir]));
		CC_left = R_left_right * CC_right + repmat(T_left_right, [1, ncols_ir*nrows_ir]);

		% homogeneous normalized camera coordinates (left)
		CC_norm_left = CC_left ./ repmat(CC_left(3, :), [3 1]);

		% homogeneous image coordinates (left)
		IC_homo_left = round(KK_rgb * CC_norm_left);
		IC_homo_left(1:2,:) = IC_homo_left(1:2,:) + 1;  % one-based coordinates

		%IDX1 = find(IC_homo_left(1,:) > 0);
		%IDX2 = find(IC_homo_left(1,:) <= ncols_rgb);
		%IDX3 = find(IC_homo_left(2,:) > 0);
		%IDX4 = find(IC_homo_left(2,:) <= nrows_rgb);
		%IDX = intersect(intersect(IDX1, IDX2), intersect(IDX3, IDX4));
		IDX = find((IC_homo_left(1,:) > 0) & (IC_homo_left(1,:) <= ncols_rgb) & (IC_homo_left(2,:) > 0) & (IC_homo_left(2,:) <= nrows_rgb));
		for ii = 1:length(IDX)
			depth1 = Img_ir_mapped(IC_homo_left(2,IDX(ii)), IC_homo_left(1,IDX(ii)));
			depth2 = CC_right(3,IDX(ii));
			%if 0 == depth1 || depth1 > depth2
				Img_ir_mapped(IC_homo_left(2,IDX(ii)), IC_homo_left(1,IDX(ii))) = depth2;
			%end;
		end;

		toc;

		rgb_output_images{kk} = Img_rgb;
		ir_output_images{kk} = Img_ir_mapped;
	end;
else
	error('improper applied method index')'
end;

%----------------------------------------------------------
% show results

for kk = 1:num_image_pairs
	img1 = rgb_output_images{kk} ./ max(max(rgb_output_images{kk}));
	img2 = ir_output_images{kk};
	img2(find(isnan(img2))) = 0;
	img2 = img2 ./ max(max(img2));
	imgBlend1 = imfuse(img1, img2, 'blend');

	figure;
	subplot(2,2,1), imshow(img1);
	subplot(2,2,2), imshow(img2);
	subplot(2,2,3), imshow(imgBlend1);
end;
