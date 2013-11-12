%----------------------------------------------------------

% at desire.kaist.ac.kr
%addpath('D:\work\sw_dev\matlab\rnd\src\statistical_analysis\kolmogorov_smirnov_test');
%addpath('E:\dataset\motion\ChaLearn_Gesture_Challenge_dataset\Sample_code_6_16_2012\Sample_code\mfunc\basic');
%cd('D:\working_copy\research_https\matlab\human_motion_analysis\THoG');

% at eden.kaist.ac.kr
addpath('E:\sangwook\work\sw_dev\matlab\rnd\src\statistical_analysis\kolmogorov_smirnov_test');
addpath('E:\sangwook\dataset\motion\ChaLearn_Gesture_Challenge_dataset\Sample_code_6_16_2012\Sample_code\mfunc\basic');
%cd('E:\sangwook\working_copy\research_https\matlab\human_motion_analysis\THoG');

% at LG notebook
%addpath('D:\sangwook\work\sw_dev\matlab\rnd\src\statistical_analysis\kolmogorov_smirnov_test');
%addpath('D:\sangwook\dataset\motion\ChaLearn_Gesture_Challenge_dataset\Sample_code_6_16_2012\Sample_code\mfunc\basic');
%cd('D:\sangwook\working_copy\research_https\matlab\human_motion_analysis\THoG');

%----------------------------------------------------------

% at desire.kaist.ac.kr
%dataset_base_directory_path = 'E:\dataset\motion\ChaLearn_Gesture_Challenge_dataset\quasi_lossless_format\train_data\';
%segmentation_info_directory_path = 'E:\dataset\motion\ChaLearn_Gesture_Challenge_dataset\Sample_code_6_16_2012\Sample_code\Examples\tempo_segment_sangwook\';

% at eden.kaist.ac.kr
dataset_base_directory_path = 'E:\sangwook\dataset\motion\ChaLearn_Gesture_Challenge_dataset\quasi_lossless_format\train_data\';
segmentation_info_directory_path = 'E:\sangwook\dataset\motion\ChaLearn_Gesture_Challenge_dataset\Sample_code_6_16_2012\Sample_code\Examples\tempo_segment_sangwook\';

% at WD external HDD
%dataset_base_directory_path = 'F:\dataset\motion\ChaLearn_Gesture_Challenge_dataset\quasi_lossless_format\train_data\';
%segmentation_info_directory_path = 'F:\dataset\motion\ChaLearn_Gesture_Challenge_dataset\Sample_code_6_16_2012\Sample_code\Examples\tempo_segment_sangwook\';

%----------------------------------------------------------

% one-shot learning.
% confusion matrix is created as a result for testing.
% scores are created as a result for testing which are evaluated using edit distance (Levenshteain distance) like ChaLearn Gesture Challenge dataset.

% two-dimensional (2D) paired Kolmogorov-Smirnov test.
% [ref] http://www.mathworks.com/matlabcentral/fileexchange/38617-two-dimensional-2d-paired-kolmogorov-smirnov-test.

dataset_idx = 20;  % 1 ~ 20.

% the number of training datasets (devel01 ~ devel20).
%  devel02: there is no information about M_45 & K_45 in ${ChaLearn_Gesture_Challenge_dataset_HOME}/data_annotation/tempo_segment.csv.
%  devel20: truth_labels{19}, which is related M_19 & K_19, has 4 labels, not but 3 labels in ${ChaLearn_Gesture_Challenge_dataset_HOME}/Sample_code_6_16_2012/Sample_code/Examples/tempo_segment/devel20.mat.
unique_gesture_count_list = [  10 10   8  10   8  10   9  11   9   9   8  11  12   8   8  13   8  10   9  9 ];
all_gesture_count_list =    [ 100 98 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 99 ];

unique_gesture_count = unique_gesture_count_list(dataset_idx);

feature_type_name = 'HoG';

feature_file_prefix = 'M_';
%feature_file_prefix = 'K_';

% two-dimensional (2D) paired Kolmogorov-Smirnov test is independent of degrees, 1deg or 10deg.
% it depends on the number of sample.
degree = '1deg';
%degree = '10deg';

feature_file_indexes = 1:47;  % M_1 ~ M_47 or K_1 ~ K_47.

%----------------------------------------------------------

feature_directory_name = sprintf('devel%02d_thog2_%s_segmented', dataset_idx, degree);

start_timestamp = datestr(clock, 30);
resultant_file_path = strcat('chalearn_', sprintf('devel%02d_', dataset_idx), feature_file_prefix, 'THoG_comparison_by_levenshtein_result_', start_timestamp, '.mat');

switch lower(degree)
    case '1deg'
        HoG_bin_width = 1;
		HoG_scale_factor = 1;
    case '10deg'
        HoG_bin_width = 10;
		HoG_scale_factor = 1;
    otherwise
        disp('Unknown degree.')
end

%----------------------------------------------------------
disp('loading HoG or THoG dataset ...');

segmentation_info = load(strcat(segmentation_info_directory_path, sprintf('devel%02d.mat', dataset_idx)));

feature_directory_path = strcat(dataset_base_directory_path, feature_directory_name, '\');

THoG_file_list = [];
for file_idx = feature_file_indexes
	file_idx_str = sprintf('%d', file_idx);

	file_list = dir(strcat(feature_directory_path, feature_file_prefix, file_idx_str, '_*.',  feature_type_name));
	THoG_file_list = [ THoG_file_list ; file_list ];
end;

num_gesture_files = length(THoG_file_list);

THoG_list = cell(1, num_gesture_files);
for kk = 1:num_gesture_files
	seq = dlmread(strcat(feature_directory_path, THoG_file_list(kk).name), ',');

	feat.filename = THoG_file_list(kk).name;
	feat.gesture = seq(1,1);  % gesture label id.
	feat.size = seq(2,1:2);
	feat.THoG = seq(3:end,:);

	THoG_list{kk} = feat;
end;
clear seq;

%----------------------------------------------------------
disp('comparison THoG using two-dimensional (2D) paired Kolmogorov-Smirnov test ...');

train_X = cell(1, unique_gesture_count);
for kk = 1:unique_gesture_count
	train_X{kk} = [];
	for frame = 1:THoG_list{kk}.size(2)
		angleData = HoG_to_angle(THoG_list{kk}.THoG(:,frame), HoG_bin_width, HoG_scale_factor);
		num_samples = length(angleData);
		train_X{kk} = [ train_X{kk} ; (frame * ones(num_samples, 1)) angleData ];  % [ frame index, angle(radian) ].
	end;
end;

predicted_gesture_labels = zeros(num_gesture_files, 1);
confusion_matrix = zeros(unique_gesture_count, unique_gesture_count);

H = zeros(1, unique_gesture_count);
pValue = zeros(1, unique_gesture_count);
KSstatistic = zeros(1, unique_gesture_count);
for kk = 1:num_gesture_files
	fprintf(1, 'comparing test dataset %d / %d ...\n', kk, num_gesture_files)

	test_X = [];
	for frame = 1:THoG_list{kk}.size(2)
		angleData = HoG_to_angle(THoG_list{kk}.THoG(:,frame), HoG_bin_width, HoG_scale_factor);
		num_samples = length(angleData);
		test_X = [ test_X ; (frame * ones(num_samples, 1)) angleData ];  % [ frame index, angle(radian) ].
	end;

	tic;
	for ii = 1:unique_gesture_count
		[H(ii), pValue(ii), KSstatistic(ii)] = kstest_2s_2d(train_X{ii}, test_X);
	end;
	fprintf(1, 'elapsed time: %f\n', toc)

	[min_KS_val, min_KS_idx] = min(KSstatistic);

	predicted_gesture_labels(kk) = THoG_list{min_KS_idx}.gesture;

	confusion_matrix(THoG_list{min_KS_idx}.gesture, THoG_list{kk}.gesture) = confusion_matrix(THoG_list{min_KS_idx}.gesture, THoG_list{kk}.gesture) + 1;

    save(resultant_file_path, 'confusion_matrix');
end;

truth_labels = segmentation_info.truth_labels;
predicted_labels = cell(length(truth_labels), 1);
idx = 1;
for kk = 1:length(truth_labels)
	labels = zeros(length(truth_labels{kk}), 1);
	for ii = 1:length(truth_labels{kk})
		labels(ii) = predicted_gesture_labels(idx);

		idx = idx + 1;
	end;

	predicted_labels{kk} = labels;
end;

[score, local_scores] = lscore(truth_labels, predicted_labels);

save(resultant_file_path, 'confusion_matrix', 'truth_labels', 'predicted_labels', 'score', 'local_scores');
