%----------------------------------------------------------
%addpath('D:\work_center\sw_dev\matlab\rnd\src\machine_vision\camera_calibration_toolbox_for_matlab\toolbox_calib');

%----------------------------------------------------------

ir_images = [
	struct('filename', 'kinect_depth_20130530T103805.png', 'rgb', false),
	struct('filename', 'kinect_depth_20130531T023152.png', 'rgb', false), 
	struct('filename', 'kinect_depth_20130531T023346.png', 'rgb', false), 
	struct('filename', 'kinect_depth_20130531T023359.png', 'rgb', false) 
];
rgb_images = [
	struct('filename', 'kinect_rgba_20130530T103805.png', 'rgb', true),
	struct('filename', 'kinect_rgba_20130531T023152.png', 'rgb', true),
	struct('filename', 'kinect_rgba_20130531T023346.png', 'rgb', true),
	struct('filename', 'kinect_rgba_20130531T023359.png', 'rgb', true)
];
num_ir_images = length(ir_images);
num_rgb_images = length(rgb_images);

%----------------------------------------------------------

if true
	% the 5th distortion parameter, kc(5) is activated.

	fc_ir = [ 5.857251103301124e+02     5.861509849627823e+02 ];  % [pixel]
	cc_ir = [ 3.360396440069350e+02     2.468430078952277e+02 ];  % [pixel]
	alpha_c_ir = [ 0.00000 ];
	kc_ir = [ -1.113144398698150e-01     3.902042354943196e-01    -2.473313414949828e-03     6.053929513996014e-03    -2.342535197486739e-01 ];

	fc_rgb = [ 5.261769128081118e+02     5.280693668967953e+02 ];  % [pixel]
	cc_rgb = [ 3.290215649965892e+02     2.651462857334770e+02 ];  % [pixel]
	alpha_c_rgb = [ 0.00000 ];
	kc_rgb = [ 2.639717236885097e-01    -9.026376922133396e-01     2.569103898876239e-03     4.773654687023216e-03     1.074728662132601e+00 ];

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
	% the 5th distortion parameter, kc(5) is deactivated.

	fc_ir = [ 5.857535922475207e+02     5.865708030703412e+02 ];  % [pixel]
	cc_ir = [ 3.351932174524685e+02     2.464165684432059e+02 ];  % [pixel]
	alpha_c_ir = [ 0.00000 ];
	kc_ir = [ -1.063901580499479e-01     3.395192881812036e-01    -2.211031053332312e-03     5.882227715342140e-03    0.0 ];

	fc_rgb = [ 5.266814231294437e+02     5.280641466171643e+02 ];  % [pixel]
	cc_rgb = [ 3.276528954184697e+02     2.652059636854492e+02 ];  % [pixel]
	alpha_c_rgb = [ 0.00000 ];
	kc_rgb = [ 2.322255151854028e-01    -5.598137839760616e-01     2.277053552942137e-03     3.720963676783346e-03     0.0 ];

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
else
	% in case of using Calib_Results_stereo.mat.

	%load('Calib_Results_ir.mat');
	%load('Calib_Results_ir_wo_k5.mat');  % when k(5) = 0

	fc_ir = fc;
	cc_ir = cc;
	alpha_c_ir = alpha_c;
	kc_ir = kc;
	ncols_ir = nx;
	nrows_ir = ny;

	%load('Calib_Results_rgba.mat');
	%load('Calib_Results_rgba_wo_k5.mat');  % when k(5) = 0

	fc_rgb = fc;
	cc_rgb = cc;
	alpha_c_rgb = alpha_c;
	kc_rgb = kc;
	ncols_rgb = nx;
	nrows_rgb = ny;
end;

%----------------------------------------------------------

% [ref]
%	${CAMEARA_CALIBRATION_TOOLBOX_FOR_MATLAB_HOME}/undistort_image.m
%	${CAMEARA_CALIBRATION_TOOLBOX_FOR_MATLAB_HOME}/rect.m
%	${CAMEARA_CALIBRATION_TOOLBOX_FOR_MATLAB_HOME}/apply_distortion.m

%----------------------------------------------------------
% load images

ir_input_images = cell(1, num_ir_images);
rgb_input_images = cell(1, num_rgb_images);
for kk = 1:num_ir_images
	% we must use double() instead of im2double().
	if ir_images(kk).rgb
		ir_input_images{kk} = double(rgb2gray(imread(ir_images(kk).filename)));
	else
		ir_input_images{kk} = double(imread(ir_images(kk).filename));
	end;
end;
for kk = 1:num_rgb_images
	if rgb_images(kk).rgb
		rgb_input_images{kk} = double(rgb2gray(imread(rgb_images(kk).filename)));
	else
		rgb_input_images{kk} = double(imread(rgb_images(kk).filename));
	end;
end;

%----------------------------------------------------------
% undistort images

ir_output_images = cell(1,num_ir_images);
for kk = 1:num_ir_images
	msg = sprintf('undistorting %s ...', ir_images(kk).filename);
	disp(msg);
	tic;

	ir_output_images{kk} = rect(ir_input_images{kk}, eye(3), fc_ir, cc_ir, kc_ir, KK_ir);

	toc;
end;

rgb_output_images = cell(1,num_rgb_images);
for kk = 1:num_rgb_images
	msg = sprintf('undistorting %s ...', rgb_images(kk).filename);
	disp(msg);
	tic;

	rgb_output_images{kk} = rect(rgb_input_images{kk}, eye(3), fc_rgb, cc_rgb, kc_rgb, KK_rgb);

	toc;
end;

%----------------------------------------------------------
% show results

for kk = 1:num_ir_images
	img = ir_output_images{kk} ./ max(max(ir_output_images{kk}));

	figure;
	imshow(img);
end;

for kk = 1:num_rgb_images
	img = rgb_output_images{kk} ./ max(max(rgb_output_images{kk}));

	figure;
	imshow(img);
end;
