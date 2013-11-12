function [ measured_accels, measured_gyros ] = load_imu_raw_data(filename, REF_GRAVITY)

fid = fopen(filename, 'r');

deg2rad = pi / 180;

% eliminate the 1st 7 lines
for ii = 1:7
	aLine = fgets(fid);
end;

data_count = 1;
while feof(fid) ~= 1
	aLine = fgets(fid);
	[SampleNum, Time, XgND, XGyro, YgND, YGyro, ZgND, ZGyro, XaND, XAccel, YaND, YAccel, ZaND, ZAccel] ...
		= strread(aLine, '%d%f%d%f%d%f%d%f%d%f%d%f%d%f', 'delimiter', ',');

	measured_accels(data_count, :) = [ XAccel, YAccel, ZAccel ] * REF_GRAVITY;
	measured_gyros(data_count, :) = [ XGyro, YGyro, ZGyro ] * deg2rad;

	data_count = data_count + 1;
end;
