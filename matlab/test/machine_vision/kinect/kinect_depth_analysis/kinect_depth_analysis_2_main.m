%----------------------------------------------------------
% The Kinect's depth output appears to be linearly proportional to the inverse of the distance to the object
% [ref]
%	http://www.ros.org/wiki/kinect_node/Calibration
%	http://stackoverflow.com/questions/8824743/kinect-raw-depth-to-distance-in-meters
%	https://groups.google.com/forum/?fromgroups=#!topic/openkinect/AxNRhG_TPHg

%----------------------------------------------------------

% at desire.kaist.ac.kr
%addpath('D:\work_center\sw_dev\matlab\rnd\src\statistical_analysis\ransac\timzaman');
%cd('D:\working_copy\research_https\matlab\kinect');

%----------------------------------------------------------

does_use_raw_depth_data = true;
used_data_file_flag = 0;  % 1, 2, 0 
image_width = 640;
image_height = 480;

depth_1_file_names = {
	'data_2/1/depth_200.txt',
	'data_2/1/depth_300.txt',
	'data_2/1/depth_400.txt',
	'data_2/1/depth_500.txt',
	'data_2/1/depth_600.txt',
	'data_2/1/depth_700.txt'
};
depth_2_file_names = {
	'data_2/2/depth_100.txt',
	'data_2/2/depth_200.txt',
	'data_2/2/depth_300.txt',
	'data_2/2/depth_400.txt',
	'data_2/2/depth_500.txt',
	'data_2/2/depth_600.txt',
	'data_2/2/depth_900.txt'
};
if 1 == used_data_file_flag
	depth_file_names = [ depth_1_file_names ];
	subplot_row_count = 2;
	subplot_col_count = 3;
elseif 2 == used_data_file_flag
	depth_file_names = [ depth_2_file_names ];
	subplot_row_count = 2;
	subplot_col_count = 4;
else
	depth_file_names = [ depth_1_file_names ; depth_2_file_names ];
	subplot_row_count = 3;
	subplot_col_count = 5;
end;

image_width = 640;
image_height = 480;

num_depth_data = length(depth_file_names);
raw_depth_data = cell(1, num_depth_data);
calc_depth_data = cell(1, num_depth_data);
for ii = 1:num_depth_data
	raw_depth_data{ii} = load(depth_file_names{ii});
	raw_depth_data{ii}(:,1:2) = raw_depth_data{ii}(:,1:2) + 1;

	% calculate depth data from raw depth data
	calc_depth_data{ii} = raw_depth_data{ii};
	%calc_depth_data{ii}(:,3) = 1 ./ raw_depth_data{ii}(:,3);
	calc_depth_data{ii}(:,3) = 1 ./ (3.33094951605675 - raw_depth_data{ii}(:,3) * 0.00307110156374373);
	%calc_depth_data{ii}(:,3) = 0.1236 * tan(raw_depth_data{ii}(:,3) / 2842.5 + 1.1863);

	calc_depth_data{ii}(isinf(calc_depth_data{ii}(:,3)),3) = 0;
end;

if does_use_raw_depth_data
	depth_data = raw_depth_data;
else
	depth_data = calc_depth_data;
end;

%----------------------------------------------------------
no = 3;  % smallest number of points required
k = 5;  % number of iterations
t = 2;  % threshold used to id a point that fits well
d = 70;  % number of nearby points required

delta_depth_data = cell(1, num_depth_data);
for ii = 1:num_depth_data
	%[pt_est, normal_est, ro_est, X_est, Y_est, Z_est, error_est, samples_used] = local_ransac_tim(depth_data{ii}, no, k, t, d);
	[normal_est, ro_est, X_est, Y_est, Z_est] = LSE_tim(depth_data{ii});

	%figure;
	%hold on;
	%%plot3(pt_est(:,1), pt_est(:,2), pt_est(:,3), 'ok')
	%mesh(X_est, Y_est, Z_est);
	%colormap([.8 .8 .8])
	%hold off;

	delta_depth_data{ii} = depth_data{ii};
	delta_depth_data{ii}(:,3) = depth_data{ii}(:,3) - (ro_est - normal_est(1).*depth_data{ii}(:,1) - normal_est(2).*depth_data{ii}(:,2)) / normal_est(3);
end;

%----------------------------------------------------------
x_range = 1:image_width;
y_range = 1:image_height;
%x_range = 220:420;
%y_range = 110:310;
figure;
for ii = 1:num_depth_data
	mesh_data = reshape(depth_data{ii}(:,3), image_height, image_width);
	%mesh_data = reshape(delta_depth_data{ii}(:,3), image_height, image_width);

	%mesh(x_range, y_range, mesh_data(y_range, x_range));
	%plot3(depth_data{ii}(:,1), depth_data{ii}(:,2), depth_data{ii}(:,3), 'o-');

	subplot(subplot_row_count, subplot_col_count, ii);
	plot(y_range, mesh_data(:,image_width/2), '-');
end;
