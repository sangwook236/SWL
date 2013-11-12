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

%total_elapsed_time = 29.46875;
%[ measured_accels, measured_gyros ] = load_imu_raw_data('adis16350_data_20100801/03_x_pos.csv', REF_GRAVITY);
%total_elapsed_time = 30.03125;
%[ measured_accels, measured_gyros ] = load_imu_raw_data('adis16350_data_20100801/04_x_neg.csv', REF_GRAVITY);
total_elapsed_time = 31.07813;
[ measured_accels, measured_gyros ] = load_imu_raw_data('adis16350_data_20100801/05_y_pos.csv', REF_GRAVITY);
%total_elapsed_time = 29.28125;
%[ measured_accels, measured_gyros ] = load_imu_raw_data('adis16350_data_20100801/06_y_neg.csv', REF_GRAVITY);
%total_elapsed_time = 30.29688;
%[ measured_accels, measured_gyros ] = load_imu_raw_data('adis16350_data_20100801/01_z_pos.csv', REF_GRAVITY);
%total_elapsed_time = 29.04688;
%[ measured_accels, measured_gyros ] = load_imu_raw_data('adis16350_data_20100801/02_z_neg.csv', REF_GRAVITY);

Nsample = size(measured_accels,1);
Ts = total_elapsed_time / Nsample;

%--------------------------------------------------------------------
% initialize gravity

initial_gravity = initialize_imu_gravity(accel_calib_param, measured_accels);

%--------------------------------------------------------------------
% get calibrated accelerations & angular rates

calibrated_accels = calculate_imu_calibrated_acceleration(accel_calib_param, measured_accels);
calibrated_gyros = calculate_imu_calibrated_angular_rate(gyro_calib_param, measured_gyros);

%--------------------------------------------------------------------
% IMU filtering by UKF

% handles to dynamic and measurement model functions
f_func = @imu_process;
g_func = @imu_observation;

% initial state and covariance
x0 = zeros(state_dim,1);
%x0(7) = -initial_gravity(1);  % a_p = initial_gravity_x
%x0(8) = -initial_gravity(2);  % a_q = initial_gravity_y
%x0(9) = -initial_gravity(3);  % a_r = initial_gravity_z
x0(10) = 1.0;  % e0 = 1.0
P0 = eye(state_dim,state_dim) * 1.0e-8;

% strengths of perturbations
Q = eye(state_dim,state_dim) * 1.0e-10;
R = eye(output_dim,output_dim) * 1.0e-10;

% initial values and space for augmented UKF (UKF3)
M = x0;
P = P0;
MM_UKF = zeros(state_dim,Nsample);
PP_UKF = zeros(state_dim,state_dim,Nsample);

params = zeros(1,11);
params(2) = Ts;
params(3) = initial_gravity(1);
params(4) = initial_gravity(2);
params(5) = initial_gravity(3);
params(6) = 10;  % beta_a_x
params(7) = 10;  % beta_a_y
params(8) = 10;  % beta_a_z
params(9) = 10;  % beta_w_x
params(10) = 10;  % beta_w_y
params(11) = 10;  % beta_w_z

alpha = 1.0e-3;  % 0.5
beta = 2;  % 2
kappa = 0;  % 3 - N

g_ix = initial_gravity(1);
g_iy = initial_gravity(2);
g_iz = initial_gravity(3);

for kk = 1:Nsample
	params(1) = kk;

	% compensate the local gravity & the earth's angular rates
	E0 = M(10);
	E1 = M(11);
	E2 = M(12);
	E3 = M(13);
	g_p = 2.0 * ((0.5 - E2*E2 - E3*E3)*g_ix + (E1*E2 + E0*E3)*g_iy + (E1*E3 - E0*E2)*g_iz);
	g_q = 2.0 * ((E1*E2 - E0*E3)*g_ix + (0.5 - E1*E1 - E3*E3)*g_iy + (E2*E3 + E0*E1)*g_iz);
	g_r = 2.0 * ((E1*E3 + E0*E2)*g_ix + (E2*E3 - E0*E1)*g_iy + (0.5 - E1*E1 - E2*E2)*g_iz);
	cc_accel = calibrated_accels(kk,:) - initial_gravity;
	cc_gyro = calibrated_gyros(kk,:);

	Y = [ cc_accel cc_gyro ]';
	[M,P,X_s,w] = ukf_predict3(M,P,f_func,Q,R,params,alpha,beta,kappa);
	[M,P] = ukf_update3(M,P,Y,g_func,R,X_s,w,[],alpha,beta,kappa);

	MM_UKF(:,kk) = M;
	PP_UKF(:,:,kk) = P;
end;

figure;
hold on;
plot(MM_UKF(1,:), 'r-');
plot(MM_UKF(2,:), 'g-');
plot(MM_UKF(3,:), 'b-');
hold off;
