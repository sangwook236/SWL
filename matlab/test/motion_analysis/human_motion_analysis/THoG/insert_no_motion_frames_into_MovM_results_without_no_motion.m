% at desire.kaist.ac.kr
dataset_base_directory_path = 'E:\dataset\motion\ChaLearn_Gesture_Challenge_dataset\quasi_lossless_format\train_data\';

% at eden.kaist.ac.kr
%dataset_base_directory_path = 'E:\sangwook\dataset\motion\ChaLearn_Gesture_Challenge_dataset\quasi_lossless_format\train_data\';

% at WD external HDD
%dataset_base_directory_path = 'F:\dataset\motion\ChaLearn_Gesture_Challenge_dataset\quasi_lossless_format\train_data\';

%----------------------------------------------------------

dataset_idx = 1;  % 1 ~ 20.
MovM_data_file_path = './chalearn_HoG_MovM/chalearn_HoG_MovM_1deg_4clusters_20130511/chalearn_devel01_M_HoG_MovM_20130430T143954.mat';
%MovM_data_file_path = './chalearn_HoG_MovM/chalearn_HoG_MovM_1deg_4clusters_20130511/chalearn_devel01_M_HoG_MovM_20130502T125432.mat';

%dataset_idx = 2;  % 1 ~ 20.
%MovM_data_file_path = './chalearn_HoG_MovM/chalearn_HoG_MovM_1deg_4clusters_20130511/chalearn_devel02_M_HoG_MovM_20130430T144005.mat';
%MovM_data_file_path = './chalearn_HoG_MovM/chalearn_HoG_MovM_1deg_4clusters_20130511/chalearn_devel02_M_HoG_MovM_20130502T125441.mat';
%MovM_data_file_path = './chalearn_HoG_MovM/chalearn_HoG_MovM_1deg_4clusters_20130511/chalearn_devel02_M_HoG_MovM_20130502T125454.mat';

%dataset_idx = 3;  % 1 ~ 20.
%MovM_data_file_path = './chalearn_HoG_MovM/chalearn_HoG_MovM_1deg_4clusters_20130511/chalearn_devel03_M_HoG_MovM_20130430T144014.mat';

%dataset_idx = 4;  % 1 ~ 20.
%MovM_data_file_path = './chalearn_HoG_MovM/chalearn_HoG_MovM_1deg_4clusters_20130511/chalearn_devel04_M_HoG_MovM_20130430T144020.mat';
%MovM_data_file_path = './chalearn_HoG_MovM/chalearn_HoG_MovM_1deg_4clusters_20130511/chalearn_devel04_M_HoG_MovM_20130502T125501.mat';

%dataset_idx = 5;  % 1 ~ 20.
%MovM_data_file_path = './chalearn_HoG_MovM/chalearn_HoG_MovM_1deg_4clusters_20130511/chalearn_devel05_M_HoG_MovM_20130430T144030.mat';
%MovM_data_file_path = './chalearn_HoG_MovM/chalearn_HoG_MovM_1deg_4clusters_20130511/chalearn_devel05_M_HoG_MovM_20130502T125530.mat';

%dataset_idx = 6;  % 1 ~ 20.
%MovM_data_file_path = './chalearn_HoG_MovM/chalearn_HoG_MovM_1deg_4clusters_20130511/chalearn_devel06_M_HoG_MovM_20130430T144044.mat';
%MovM_data_file_path = './chalearn_HoG_MovM/chalearn_HoG_MovM_1deg_4clusters_20130511/chalearn_devel06_M_HoG_MovM_20130502T125547.mat';

%dataset_idx = 7;  % 1 ~ 20.
%MovM_data_file_path = './chalearn_HoG_MovM/chalearn_HoG_MovM_1deg_4clusters_20130511/chalearn_devel07_M_HoG_MovM_20130430T144128.mat';
%MovM_data_file_path = './chalearn_HoG_MovM/chalearn_HoG_MovM_1deg_4clusters_20130511/chalearn_devel07_M_HoG_MovM_20130502T125600.mat';

%dataset_idx = 8;  % 1 ~ 20.
%MovM_data_file_path = './chalearn_HoG_MovM/chalearn_HoG_MovM_1deg_4clusters_20130511/chalearn_devel08_M_HoG_MovM_20130427T234952.mat';
%MovM_data_file_path = './chalearn_HoG_MovM/chalearn_HoG_MovM_1deg_4clusters_20130511/chalearn_devel08_M_HoG_MovM_20130430T144145.mat';
%MovM_data_file_path = './chalearn_HoG_MovM/chalearn_HoG_MovM_1deg_4clusters_20130511/chalearn_devel08_M_HoG_MovM_20130502T125615.mat';

%dataset_idx = 9;  % 1 ~ 20.
%MovM_data_file_path = './chalearn_HoG_MovM/chalearn_HoG_MovM_1deg_4clusters_20130511/chalearn_devel09_M_HoG_MovM_20130430T144157.mat';
%MovM_data_file_path = './chalearn_HoG_MovM/chalearn_HoG_MovM_1deg_4clusters_20130511/chalearn_devel09_M_HoG_MovM_20130502T125629.mat';

%dataset_idx = 10;  % 1 ~ 20.
%MovM_data_file_path = './chalearn_HoG_MovM/chalearn_HoG_MovM_1deg_4clusters_20130511/chalearn_devel10_M_HoG_MovM_20130430T144207.mat';
%MovM_data_file_path = './chalearn_HoG_MovM/chalearn_HoG_MovM_1deg_4clusters_20130511/chalearn_devel10_M_HoG_MovM_20130502T125637.mat';

feature_file_indexes = 1:47;  % M_1 ~ M_47 or K_1 ~ K_47.

%----------------------------------------------------------

MovMs = load(MovM_data_file_path);
%MovMs_orig = MovMs;

feature_directory_name = sprintf('devel%02d_thog2/', dataset_idx);

for file_idx = feature_file_indexes
	no_motion_file_path = strcat(dataset_base_directory_path, feature_directory_name, sprintf('M_%d_no_motion.txt', file_idx));

	fid = fopen(no_motion_file_path);
	no_motion_frame_indexes = textscan(fid, '%d');
	fclose(fid);

	no_motion_frame_indexes = no_motion_frame_indexes{1};

	if ~isempty(no_motion_frame_indexes)
		for kk = no_motion_frame_indexes'
			ff = double(kk);
			MovMs.Mu{file_idx} = [ MovMs.Mu{file_idx}(:,1:(ff-1)) zeros(size(MovMs.Mu{file_idx}, 1), 1) MovMs.Mu{file_idx}(:,ff:end) ];
			MovMs.Kappa{file_idx} = [ MovMs.Kappa{file_idx}(:,1:(ff-1)) zeros(size(MovMs.Kappa{file_idx}, 1), 1) MovMs.Kappa{file_idx}(:,ff:end) ];
			MovMs.Alpha{file_idx} = [ MovMs.Alpha{file_idx}(:,1:(ff-1)) zeros(size(MovMs.Alpha{file_idx}, 1), 1) MovMs.Alpha{file_idx}(:,ff:end) ];
		end;
	end;
end;

Mu = MovMs.Mu;
Kappa = MovMs.Kappa;
Alpha = MovMs.Alpha;
save(MovM_data_file_path, 'Mu', 'Kappa', 'Alpha');
