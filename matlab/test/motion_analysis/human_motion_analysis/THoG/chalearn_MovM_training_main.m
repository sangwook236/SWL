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
%dataset_base_directory_path = 'E:\dataset\motion\ChaLearn_Gesture_Challenge_dataset\quasi_lossless_format\train_data\';

% at eden.kaist.ac.kr
dataset_base_directory_path = 'E:\sangwook\dataset\motion\ChaLearn_Gesture_Challenge_dataset\quasi_lossless_format\train_data\';

% at WD external HDD
%dataset_base_directory_path = 'F:\dataset\motion\ChaLearn_Gesture_Challenge_dataset\quasi_lossless_format\train_data\';

%----------------------------------------------------------

dataset_directory_name = 'devel01';
feature_directory_name = 'devel01_thog2';

%feature_type_name = 'THoG';
feature_type_name = 'HoG';

feature_file_prefix = 'M_';
%feature_file_prefix = 'K_';

degree = '1deg';
%degree = '10deg';

num_clusters = 4;

does_use_random_initialization = true;
does_save_labels = false;

% HoG_sequence_indexes(1):HoG_frame_indexes(1) ~ HoG_sequence_indexes(2):HoG_frame_indexes(2)
% if index == 0, it means the first or last sequence/frame index.
HoG_sequence_indexes = [ 0 0 ];
HoG_frame_indexes = [ 0 0 ];

max_step = 1000;
tol = 1e-3;

%----------------------------------------------------------
disp('loading HoG or THoG dataset ...');

start_timestamp = datestr(clock, 30);
[ trainSeqs trainLabels testSeqs testLabels ] = chalearn_load_dataset(dataset_base_directory_path, dataset_directory_name, feature_directory_name, feature_file_prefix, strcat('.', feature_type_name));
resultant_label_file_path = strcat('chalearn_', dataset_directory_name, '_label', '.mat');
resultant_MovM_file_path = strcat('chalearn_', dataset_directory_name, '_', feature_file_prefix, feature_type_name, '_MovM_', start_timestamp, '.mat');

seqs = [ trainSeqs testSeqs ];
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

if does_save_labels
    Labels = [ trainLabels testLabels ];
    TrainSequenceCount = size(trainSeqs, 2);
    save(resultant_label_file_path, 'Labels', 'TrainSequenceCount');

    clear Labels, TrainSequenceCount;
end;

clear trainSeqs trainLabels testSeqs testLabels;

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
