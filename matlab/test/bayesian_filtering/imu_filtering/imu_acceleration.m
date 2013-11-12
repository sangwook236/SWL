% [ref]
%	"Image Deblurring using Inertial Measurement Sensors", by Neel Joshi, Sing Bing Kang, C. Lawrence Zitnick, & Richard Szeliski,
%	ACM Transactions on Graphics, 2010

%addpath('E:\work_center\sw_dev\matlab\rnd\bayesian_filtering\ekfukf_1_2\ekfukf');

%--------------------------------------------------------------------
% reference gravity & angular velocity

deg2rad = pi / 180.0;
phi = 36.368 * deg2rad;  % latitude [rad]
lambda = 127.364 * deg2rad;  % longitude [rad]
h = 71.0;  % altitude: 71 ~ 82 [m]
sin_phi = sin(phi);
sin_2phi = sin(2 * phi);

% [ref] wikipedia
% (latitude, longitude, altitude) = (phi, lambda, h) = (36.36800, 127.35532, ?)
% g(phi, h) = 9.780327 * (1 + 0.0053024 * sin(phi)^2 - 0.0000058 * sin(2 * phi)^2) - 3.086 * 10^-6 * h
REF_GRAVITY = 9.780327 * (1.0 + 0.0053024 * sin_phi*sin_phi - 0.0000058 * sin_2phi*sin_2phi) - 3.086e-6 * h;  % [m/sec^2]

% [ref] "The Global Positioning System and Inertial Navigation", Jay Farrell & Mattthew Barth, pp. 22
REF_ANGULAR_VEL = 7.292115e-5;  % [rad/sec]

state_dim = 22;
output_dim = 6;

%--------------------------------------------------------------------
% calibration parameters

accel_calib_param = [
	0.581636  -0.205962  0.453323  2.68571e-005  -0.000488519  0.000573955  -0.00581539  0.0158689  -0.000389812
];
accel_calib_covar_param = [
	7.67441e-005  9.83051e-007  3.93396e-007  4.45687e-008  -6.72968e-008  -4.45194e-009  8.61712e-007  -5.29287e-006  -6.30373e-007
	9.83051e-007  3.58234e-005  -6.09996e-007  -4.03654e-009  -2.62386e-008  -5.35809e-009  -7.73065e-008  1.29453e-006  -3.15509e-007
	3.93396e-007  -6.09996e-007  1.4242e-005  -2.43596e-007  5.44294e-008  1.96181e-008  -1.17906e-008  6.60122e-006  7.73491e-008
	4.45687e-008  -4.03654e-009  -2.43596e-007  1.43817e-006  9.32452e-010  -1.09994e-007  1.64944e-008  -2.63846e-006  -3.33297e-008
	-6.72968e-008  -2.62386e-008  5.44294e-008  9.32452e-010  8.01742e-007  -6.08574e-008  -4.858e-005  4.97596e-005  1.53393e-008
	-4.45194e-009  -5.35809e-009  1.96181e-008  -1.09994e-007  -6.08574e-008  1.88039e-007  6.61142e-009  -2.6015e-007  1.36527e-008
	8.61712e-007  -7.73065e-008  -1.17906e-008  1.64944e-008  -4.858e-005  6.61142e-009  0.00841425  -0.00841453  -2.27404e-008
	-5.29287e-006  1.29453e-006  6.60122e-006  -2.63846e-006  4.97596e-005  -2.6015e-007  -0.00841453  0.00852298  1.90004e-006
	-6.30373e-007  -3.15509e-007  7.73491e-008  -3.33297e-008  1.53393e-008  1.36527e-008  -2.27404e-008  1.90004e-006  2.12793e-006
];
gyro_calib_param = [
	0.000876087  -0.0034884  0.00222278
];
gyro_calib_covar_param = [
	1.30834e-007  3.57487e-009  8.74685e-009
	3.57487e-009  6.45255e-008  -4.28119e-008
	8.74685e-009  -4.28119e-008  5.25857e-007
];

%--------------------------------------------------------------------
% load measured raw data

dataset_type = 'x';  % 'x', 'y', 'z', 't'
dataset_index = 1;  % 1, 2, 3

if strcmp(dataset_type, 'x') == 1
	if dataset_index == 1
		total_elapsed_time = 12.53125;
		[ measured_accels, measured_gyros ] = load_imu_raw_data('adis16350_data_20100903/x_neg_50cm_40msec_1.csv', REF_GRAVITY);
	elseif dataset_index == 2
		total_elapsed_time = 12.45313;
		[ measured_accels, measured_gyros ] = load_imu_raw_data('adis16350_data_20100903/x_neg_50cm_40msec_2.csv', REF_GRAVITY);
	elseif dataset_index == 3
		total_elapsed_time = 12.5;
		[ measured_accels, measured_gyros ] = load_imu_raw_data('adis16350_data_20100903/x_neg_50cm_40msec_3.csv', REF_GRAVITY);
	else
		disp('dataset index error !!!');
	end;
elseif strcmp(dataset_type, 'y') == 1
	if dataset_index == 1
		total_elapsed_time = 12.5;
		[ measured_accels, measured_gyros ] = load_imu_raw_data('adis16350_data_20100903/y_pos_50cm_40msec_1.csv', REF_GRAVITY);
	elseif dataset_index == 2
		total_elapsed_time = 12.54688;
		[ measured_accels, measured_gyros ] = load_imu_raw_data('adis16350_data_20100903/y_pos_50cm_40msec_2.csv', REF_GRAVITY);
	elseif dataset_index == 3
		total_elapsed_time = 12.46875;
		[ measured_accels, measured_gyros ] = load_imu_raw_data('adis16350_data_20100903/y_pos_50cm_40msec_3.csv', REF_GRAVITY);
	else
		disp('dataset index error !!!');
	end;
elseif strcmp(dataset_type, 'z') == 1
	if dataset_index == 1
		total_elapsed_time = 12.46875;
		[ measured_accels, measured_gyros ] = load_imu_raw_data('adis16350_data_20100903/z_neg_50cm_40msec_1.csv', REF_GRAVITY);
	elseif dataset_index == 2
		total_elapsed_time = 12.54688;
		[ measured_accels, measured_gyros ] = load_imu_raw_data('adis16350_data_20100903/z_neg_50cm_40msec_2.csv', REF_GRAVITY);
	elseif dataset_index == 3
		total_elapsed_time = 12.54688;
		[ measured_accels, measured_gyros ] = load_imu_raw_data('adis16350_data_20100903/z_neg_50cm_40msec_3.csv', REF_GRAVITY);
	else
		disp('dataset index error !!!');
	end;
elseif strcmp(dataset_type, 't') == 1
	if dataset_index == 1
		total_elapsed_time = 12.45313;
		[ measured_accels, measured_gyros ] = load_imu_raw_data('adis16350_data_20100903/tilt_50cm_40msec_1.csv', REF_GRAVITY);
	elseif dataset_index == 2
		total_elapsed_time = 12.39063;
		[ measured_accels, measured_gyros ] = load_imu_raw_data('adis16350_data_20100903/tilt_50cm_40msec_2.csv', REF_GRAVITY);
	elseif dataset_index == 3
		total_elapsed_time = 12.48438;
		[ measured_accels, measured_gyros ] = load_imu_raw_data('adis16350_data_20100903/tilt_50cm_40msec_3.csv', REF_GRAVITY);
	else
		disp('dataset index error !!!');
	end;
else
	disp('dataset type error !!!');
end;

Nsample = size(measured_accels,1);
Ts = total_elapsed_time / Nsample;

%--------------------------------------------------------------------
% get calibrated accelerations & angular rates

calibrated_accels = calculate_imu_calibrated_acceleration(accel_calib_param, measured_accels);
calibrated_gyros = calculate_imu_calibrated_angular_rate(gyro_calib_param, measured_gyros);

%--------------------------------------------------------------------
% initialize gravity

if strcmp(dataset_type, 'x') == 1
	total_elapsed_time = 41.625;
	[ initial_accels, initial_gyros ] = load_imu_raw_data('adis16350_data_20100903/x_neg_initial_40msec.csv', REF_GRAVITY);
elseif strcmp(dataset_type, 'y') == 1
	total_elapsed_time = 41.5;
	[ initial_accels, initial_gyros ] = load_imu_raw_data('adis16350_data_20100903/y_pos_initial_40msec.csv', REF_GRAVITY);
elseif strcmp(dataset_type, 'z') == 1
	total_elapsed_time = 41.46875;
	[ initial_accels, initial_gyros ] = load_imu_raw_data('adis16350_data_20100903/z_neg_initial_40msec.csv', REF_GRAVITY);
elseif strcmp(dataset_type, 't') == 1
	total_elapsed_time = 41.32813;
	[ initial_accels, initial_gyros ] = load_imu_raw_data('adis16350_data_20100903/tilt_initial_40msec.csv', REF_GRAVITY);
else
	disp('dataset type error !!!');
end;

calibrated_initial_gravity = initialize_imu_gravity(accel_calib_param, initial_accels);

%--------------------------------------------------------------------
%

% the local gravity in the initial frame
g_i = calibrated_initial_gravity';

% rotation matrix from the initial frame to the current frame at time t
R_t_i = eye(3);
% current angular position in the initial frame
theta_i_t = zeros(3,1);
% current position in the initial frame
p_i_t = zeros(3,1);
% current velocity in the initial frame
v_i_t = zeros(3,1);
% current acceleration in the initial frame
% FIXME [check] >>
a_i_t = g_i;
% current angular velocity in the current frame
w_t_t = zeros(3,1);
% current acceleration in the current frame
a_t_t = zeros(3,1);

for nn = 1:Nsample
	p_i_t = p_i_t + v_i_t * Ts + 0.5 * (a_i_t - g_i) * Ts^2;
	v_i_t = v_i_t + (a_i_t - g_i) * Ts;

	theta_i_t = (R_t_i' * w_t_t) * Ts + theta_i_t;
	% FIXME [check] >>
	R_t_i = rot(theta_i_t, norm(theta_i_t));

	w_t_t = calibrated_gyros(nn,:)';
	a_t_t = calibrated_accels(nn,:)';
	a_i_t = R_t_i' * a_t_t;
end;
