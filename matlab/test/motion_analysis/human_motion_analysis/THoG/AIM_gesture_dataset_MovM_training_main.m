%----------------------------------------------------------

% at desire.kaist.ac.kr
%addpath('D:\work\sw_dev\matlab\rnd\src\statistical_analysis\circstat\CircStat2012a');
%addpath('D:\work\sw_dev\matlab\rnd\src\statistical_analysis\movmf\vmfmatlab');
%addpath('D:\working_copy\swl_https\matlab\src\statistical_analysis\directional_statistics');
%addpath('D:\working_copy\swl_https\matlab\src\statistical_inference\em');
%cd('D:\working_copy\research_https\matlab\human_motion_analysis\THoG');

% at eden.kaist.ac.kr
addpath('E:\sangwook\work\sw_dev\matlab\rnd\src\statistical_analysis\circstat\CircStat2012a');
addpath('E:\sangwook\work\sw_dev\matlab\rnd\src\statistical_analysis\movmf\vmfmatlab')
addpath('E:\sangwook\working_copy\swl_https\matlab\src\statistical_analysis\directional_statistics');
addpath('E:\sangwook\working_copy\swl_https\matlab\src\statistical_inference\em');
%cd('E:\sangwook\working_copy\research_https\matlab\human_motion_analysis\THoG');

% at LG notebook
%addpath('D:\sangwook\work\sw_dev\cpp\rnd\src\probabilistic_graphical_model\crf\hcrf\HCRF2.0b\bin\openMP');
%addpath('D:\sangwook\work\sw_dev\matlab\rnd\src\statistical_analysis\circstat\CircStat2012a');
%addpath('D:\sangwook\work\sw_dev\matlab\rnd\src\statistical_analysis\movmf\vmfmatlab')
%addpath('D:\sangwook\working_copy\swl_https\matlab\src\statistical_analysis\directional_statistics');
%addpath('D:\sangwook\working_copy\swl_https\matlab\src\statistical_inference\em');
%cd('D:\sangwook\working_copy\research_https\matlab\human_motion_analysis\THoG');

%----------------------------------------------------------

% at desire.kaist.ac.kr
%dataset_base_directory_path = 'F:\AIM_gesture_dataset\';
%dataset_base_directory_path = 'F:\AIM_gesture_dataset_segmented\';

% at eden.kaist.ac.kr
%dataset_base_directory_path = 'E:\sangwook\AIM_gesture_dataset\';
dataset_base_directory_path = 'E:\sangwook\AIM_gesture_dataset_segmented\';

% at WD external HDD
%dataset_base_directory_path = 'F:\AIM_gesture_dataset\';
%dataset_base_directory_path = 'F:\AIM_gesture_dataset_segmented\';

%----------------------------------------------------------

%feature_directory_name = 's01_sangwook_lee_20120719_per_gesture_avi_640x480_30fps_3000kbps_thog2';
feature_directory_name = 's01_sangwook_lee_20120719_per_gesture_mp4_640x480_thog';
feature_file_list_file_name = 'file_list_s01_thog.txt';

resultant_file_header_name = 'AIM_gesture_dataset_s01_sangwook_lee';
%feature_type_name = 'THoG';
feature_type_name = 'HoG';

degree = '1deg';
%degree = '10deg';

does_use_random_initialization = true;

% HoG_sequence_indexes(1):HoG_frame_indexes(1) ~ HoG_sequence_indexes(2):HoG_frame_indexes(2)
% if index == 0, it means the first or last sequence/frame index.
HoG_sequence_indexes = [ 0 0 ];
HoG_frame_indexes = [ 0 0 ];

num_clusters = 4;
max_step = 1000;
tol = 1e-3;

%----------------------------------------------------------
disp('loading HoG or THoG dataset ...');

start_timestamp = datestr(clock, 30);
resultant_MovM_file_path = strcat(resultant_file_header_name, '_', feature_type_name, '_MovM_', start_timestamp, '.mat');

% load HoG or THoG dataset
[ seqs ] = AIM_gesture_dataset_load_dataset(dataset_base_directory_path, feature_directory_name, feature_file_list_file_name, strcat('.', feature_type_name));

switch lower(degree)
    case '1deg'
        HoG_sequences = seqs;
        HoG_bin_width = 1;  % 1 deg.
        HoG_scale_factor = 2;
    case '10deg'
        HoG_sequences = cell(size(seqs));
        for ii = 1:length(seqs)
            HoG_sequences{ii} = zeros(36, size(seqs{ii}, 2));
            for jj = 1:36
                HoG_sequences{ii}(jj, :) = sum(seqs{ii}(((jj-1)*10+1):(jj*10), :), 1);
            end;
        end;
        HoG_bin_width = 10;  % 10 deg.
        HoG_scale_factor = 2;
    otherwise
        disp('Unknown degree.')
end
clear seqs;

%----------------------------------------------------------
disp('training MovM from HoG or THoG ...');

if does_use_random_initialization
    init_mu = [];
    init_kappa = [];
    init_alpha = [];
else
    % use equally spaced values
    tmp_mu = 0 + (2*pi - 0) * rand;
    tmp_mu_spacing = 2*pi / num_clusters;
	init_mu = mod((0:(num_clusters-1)) * tmp_mu_spacing + tmp_mu, 2*pi);
	tmp_kappa = 0.5 + (5.0 - 0.5) * rand;
	init_kappa = ones(1, num_clusters) * tmp_kappa;
	init_alpha = ones(1, num_clusters) / num_clusters;
end;

tic;
[ Mu, Kappa, Alpha, IterationStep ] = train_MovM_from_HoG(HoG_sequences, HoG_sequence_indexes, HoG_frame_indexes, HoG_bin_width, HoG_scale_factor, num_clusters, init_mu, init_kappa, init_alpha, max_step, tol, resultant_MovM_file_path);
toc;
