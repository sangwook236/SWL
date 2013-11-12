function calibrated_initial_gravity = initialize_imu_gravity(accel_calib_param, measured_accels)

data_count = size(measured_accels, 1);

sum = zeros(1, 3);

for ii = 1:data_count
	calibrated_accel = calculate_imu_calibrated_acceleration(accel_calib_param, measured_accels(ii,:));
	sum = sum + calibrated_accel;
end;

calibrated_initial_gravity =  sum / data_count;
