function calibrated_gyros = calculate_imu_calibrated_angular_rate(gyro_calib_param, measured_gyros)

b_wx = gyro_calib_param(1);
b_wy = gyro_calib_param(2);
b_wz = gyro_calib_param(3);

data_count = size(measured_gyros,1);
calibrated_gyros = zeros(data_count,3);

for ii = 1:data_count
	l_wx = measured_gyros(ii,1);
	l_wy = measured_gyros(ii,2);
	l_wz = measured_gyros(ii,3);

	w_x = l_wx - b_wx;
	w_y = l_wy - b_wy;
	w_z = l_wz - b_wz;

	calibrated_gyros(ii,:) = [ w_x w_y w_z ];
end;
