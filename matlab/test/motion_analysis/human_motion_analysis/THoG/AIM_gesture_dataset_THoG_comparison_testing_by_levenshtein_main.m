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
%dataset_base_directory_path = 'F:\AIM_gesture_dataset_segmented\';

% at eden.kaist.ac.kr
dataset_base_directory_path = 'E:\sangwook\AIM_gesture_dataset_segmented\';

% at WD external HDD
%dataset_base_directory_path = 'F:\AIM_gesture_dataset_segmented\';

%----------------------------------------------------------

% one-shot learning.
% confusion matrix is created as a result for testing.
% scores are created as a result for testing which are evaluated using edit distance (Levenshteain distance) like ChaLearn Gesture Challenge dataset.

% two-dimensional (2D) paired Kolmogorov-Smirnov test.
% [ref] http://www.mathworks.com/matlabcentral/fileexchange/38617-two-dimensional-2d-paired-kolmogorov-smirnov-test.

training_gesture_file_index_list = [ 1 17 33 49 65 81 97 109 121 136 151 166 181 196 211 226 241 256 ];

dataset_file_list_path = strcat(dataset_base_directory_path, 's01_sangwook_lee_20120719_per_gesture_mp4_640x480_thog\file_list_s01_thog.txt');
dataset_label_list_path = strcat(dataset_base_directory_path, 's01_sangwook_lee_20120719_per_gesture_mp4_640x480_thog\file_list_s01_thog_label.txt');

feature_type_name = 'HoG';
%feature_type_name = 'THoG';

% two-dimensional (2D) paired Kolmogorov-Smirnov test is independent of degrees, 1deg or 10deg.
% it depends on the number of sample.
degree = '1deg';
%degree = '10deg';

%testing_gesture_file_index_list = 1:num_gesture_files;  % 1 ~ 270.
testing_gesture_file_index_list = 1:270;  % 1 ~ 270.

%----------------------------------------------------------

feature_directory_name = sprintf('s01_sangwook_lee_20120719_per_gesture_mp4_640x480_thog_%s_hcrf', degree);

start_timestamp = datestr(clock, 30);
resultant_file_path = strcat('AIM_gesture_datasets_s01_THoG_comparison_by_levenshtein_result_', start_timestamp, '.mat');

switch lower(degree)
    case '1deg'
        HoG_bin_width = 1;
		%HoG_scale_factor = 1.0;  % 45 mins per testing.
		%HoG_scale_factor = 0.5;  % 6 mins per testing.
		HoG_scale_factor = 0.2;  % 60 secs per testing.
		%HoG_scale_factor = 0.1;  % 15 secs per testing.
    case '10deg'
        HoG_bin_width = 10;
		%HoG_scale_factor = 1.0;  % 45 mins per testing.
		%HoG_scale_factor = 0.5;  % 6 mins per testing.
		HoG_scale_factor = 0.2;  % 60 secs per testing.
		%HoG_scale_factor = 0.1;  % 15 secs per testing.
    otherwise
        disp('Unknown degree.')
end

%----------------------------------------------------------
disp('loading HoG or THoG dataset ...');

training_gesture_count = length(training_gesture_file_index_list);

fid = fopen(dataset_file_list_path);
dataset_file_list = textscan(fid, '%s');
fclose(fid);
THoG_file_list = dataset_file_list{1};
THoG_label_list = dlmread(dataset_label_list_path);

feature_directory_path = strcat(dataset_base_directory_path, feature_directory_name, '\');

num_gesture_files = length(THoG_file_list);

THoG_list = cell(1, num_gesture_files);
for kk = 1:num_gesture_files
	seq = dlmread(strcat(feature_directory_path, THoG_file_list{kk}, '.', feature_type_name), ',');

	feat.filename = strcat(THoG_file_list{kk}, '.', feature_type_name);
	feat.gesture = THoG_label_list(kk);  % gesture label id.
	feat.size = seq(1,1:2);
	feat.THoG = seq(2:end,:);

	THoG_list{kk} = feat;
end;
clear seq;

%----------------------------------------------------------
disp('comparison THoG using two-dimensional (2D) paired Kolmogorov-Smirnov test ...');

train_X = cell(1, training_gesture_count);
for kk = 1:training_gesture_count
	train_X{kk} = [];
	for frame = 1:THoG_list{training_gesture_file_index_list(kk)}.size(2)
		angleData = HoG_to_angle(THoG_list{training_gesture_file_index_list(kk)}.THoG(:,frame), HoG_bin_width, HoG_scale_factor);
		num_samples = length(angleData);
		train_X{kk} = [ train_X{kk} ; (frame * ones(num_samples, 1)) angleData ];  % [ frame index, angle(radian) ].
	end;
end;

predicted_gesture_labels = zeros(num_gesture_files, 1);
confusion_matrix = zeros(training_gesture_count, training_gesture_count);

H = zeros(1, training_gesture_count);
pValue = zeros(1, training_gesture_count);
KSstatistic = zeros(1, training_gesture_count);
%for kk = 1:num_gesture_files
for kk = testing_gesture_file_index_list
	fprintf(1, 'comparing test dataset %d / %d ...\n', kk, num_gesture_files)

	test_X = [];
	for frame = 1:THoG_list{kk}.size(2)
		angleData = HoG_to_angle(THoG_list{kk}.THoG(:,frame), HoG_bin_width, HoG_scale_factor);
		num_samples = length(angleData);
		test_X = [ test_X ; (frame * ones(num_samples, 1)) angleData ];  % [ frame index, angle(radian) ].
	end;

	tic;
	for ii = 1:training_gesture_count
		[H(ii), pValue(ii), KSstatistic(ii)] = kstest_2s_2d(train_X{ii}, test_X);
	end;
	fprintf(1, 'elapsed time: %f\n', toc)

	[min_KS_val, min_KS_idx] = min(KSstatistic);

	%pred_gest_idx = THoG_list{min_KS_idx}.gesture;
	pred_gest_idx = THoG_list{training_gesture_file_index_list(min_KS_idx)}.gesture;
	true_gest_idx = THoG_list{kk}.gesture;

	predicted_gesture_labels(kk) = pred_gest_idx;

	confusion_matrix(pred_gest_idx, true_gest_idx) = confusion_matrix(pred_gest_idx, true_gest_idx) + 1;

    save(resultant_file_path, 'confusion_matrix');
end;

truth_labels = cell(length(THoG_label_list), 1);
predicted_labels = cell(length(truth_labels), 1);
for kk = 1:length(truth_labels)
	truth_labels{kk} = THoG_label_list(kk);
	predicted_labels{kk} = predicted_gesture_labels(kk);
end;

[score, local_scores] = lscore(truth_labels, predicted_labels);

save(resultant_file_path, 'confusion_matrix', 'truth_labels', 'predicted_labels', 'score', 'local_scores');
