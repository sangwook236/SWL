function calibrated_accels = calculate_imu_calibrated_acceleration(accel_calib_param, measured_accels)

b_gx = accel_calib_param(1);
b_gy = accel_calib_param(2);
b_gz = accel_calib_param(3);
s_gx = accel_calib_param(4);
s_gy = accel_calib_param(5);
s_gz = accel_calib_param(6);
theta_gyz = accel_calib_param(7);
theta_gzx = accel_calib_param(8);
theta_gzy = accel_calib_param(9);

tan_gyz = tan(theta_gyz);
tan_gzx = tan(theta_gzx);
tan_gzy = tan(theta_gzy);
cos_gyz = cos(theta_gyz);
cos_gzx = cos(theta_gzx);
cos_gzy = cos(theta_gzy);

data_count = size(measured_accels,1);
calibrated_accels = zeros(data_count,3);

for ii = 1:data_count
	l_gx = measured_accels(ii,1);
	l_gy = measured_accels(ii,2);
	l_gz = measured_accels(ii,3);
	
	g_x = (l_gx - b_gx) / (1.0 + s_gx);
	g_y = tan_gyz * (l_gx - b_gx) / (1.0 + s_gx) + (l_gy - b_gy) / ((1.0 + s_gy) * cos_gyz);
	g_z = (tan_gzx * tan_gyz - tan_gzy / cos_gzx) * (l_gx - b_gx) / (1.0 + s_gx) + ((l_gy - b_gy) * tan_gzx) / ((1.0 + s_gy) * cos_gyz) + (l_gz - b_gz) / ((1.0 + s_gz) * cos_gzx * cos_gzy);

	calibrated_accels(ii,:) = [ g_x g_y g_z ];
end;
