%----------------------------------------------------------

% at desire.kaist.ac.kr
%addpath('D:\work\sw_dev\cpp\rnd\src\probabilistic_graphical_model\crf\hcrf\HCRF2.0b\bin\openMP');
%addpath('D:\work\sw_dev\cpp\rnd\src\probabilistic_graphical_model\crf\hcrf\HCRF2.0b\samples\matlab');
%addpath('E:\dataset\motion\ChaLearn_Gesture_Challenge_dataset\Sample_code_6_16_2012\Sample_code\mfunc\basic');
%cd('D:\working_copy\research_https\matlab\human_motion_analysis\THoG');

% at eden.kaist.ac.kr
addpath('E:\sangwook\work\sw_dev\cpp\rnd\src\probabilistic_graphical_model\crf\hcrf\HCRF2.0b\bin\openMP');
addpath('E:\sangwook\work\sw_dev\cpp\rnd\src\probabilistic_graphical_model\crf\hcrf\HCRF2.0b\samples\matlab');
addpath('E:\sangwook\dataset\motion\ChaLearn_Gesture_Challenge_dataset\Sample_code_6_16_2012\Sample_code\mfunc\basic');
%cd('E:\sangwook\working_copy\research_https\matlab\human_motion_analysis\THoG');

% at LG notebook
%addpath('D:\sangwook\work\sw_dev\cpp\rnd\src\probabilistic_graphical_model\crf\hcrf\HCRF2.0b\bin\openMP');
%addpath('D:\sangwook\work\sw_dev\cpp\rnd\src\probabilistic_graphical_model\crf\hcrf\HCRF2.0b\samples\matlab');
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
%segmentation_info_directory_path = 'F:\sangwook\dataset\motion\ChaLearn_Gesture_Challenge_dataset\Sample_code_6_16_2012\Sample_code\Examples\tempo_segment_sangwook\';

%----------------------------------------------------------

% one-shot learning.
% scores are as a result for testing created which are evaluated using edit distance (Levenshteain distance) like ChaLearn Gesture Challenge dataset.

dataset_idx = 1;  % 1 ~ 20.

% the number of training datasets (devel01 ~ devel20).
%  devel02: there is no information about M_45 & K_45 in ${ChaLearn_Gesture_Challenge_dataset_HOME}/data_annotation/tempo_segment.csv.
%  devel20: truth_labels{19}, which is related M_19 & K_19, has 4 labels, not but 3 labels in ${ChaLearn_Gesture_Challenge_dataset_HOME}/Sample_code_6_16_2012/Sample_code/Examples/tempo_segment/devel20.mat.
unique_gesture_count_list = [  10 10   8  10   8  10   9  11   9   9   8  11  12   8   8  13   8  10   9  9 ];
all_gesture_count_list =    [ 100 98 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 99 ];

unique_gesture_count = unique_gesture_count_list(dataset_idx);

%feature_directory_name = sprintf('devel%02d_thog2_1deg_segmented', dataset_idx);
feature_directory_name = sprintf('devel%02d_thog2_10deg_segmented', dataset_idx);

feature_type_name = 'HoG';

feature_file_prefix = 'M_';
%feature_file_prefix = 'K_';

feature_file_indexes = 1:47;  % M_1 ~ M_47 or K_1 ~ K_47.

does_train_CRF = true;
does_train_HCRF = true;
does_train_LDCRF = true;

regularization_L2 = 10;
regularization_L1 = 10;
regularization_L1_feature_types = 'ALL';  % ALL, NODE, EDGE.
%optimizer = 'bfgs';  % cpu low, memory high.
optimizer = 'lbfgs';  % cpu high, memory low.
max_iterations = 300;

%num_hidden_states_list = 7:12;
num_hidden_states_list = 9;
%window_size_list = 11:15;
window_size_list = 8;

%----------------------------------------------------------
disp('loading HoG or THoG dataset ...');

start_timestamp = datestr(clock, 30);
resultant_file_path_prefix = strcat('chalearn_', sprintf('devel%02d_', dataset_idx), feature_file_prefix, 'HCRF_testing_by_levenshtein_result_');

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

trainCompleteSeqs = {};
trainCompleteLabels = {};
allCompleteSeqs = {};
allCompleteLabels = {};

for kk = 1:num_gesture_files
	if kk <= unique_gesture_count
		trainCompleteSeqs = [ trainCompleteSeqs THoG_list{kk}.THoG ];
		trainCompleteLabels = [ trainCompleteLabels (THoG_list{kk}.gesture * ones(1, size(THoG_list{kk}.THoG, 2))) ];
	end;

	allCompleteSeqs = [ allCompleteSeqs THoG_list{kk}.THoG ];
	allCompleteLabels = [ allCompleteLabels (THoG_list{kk}.gesture * ones(1, size(THoG_list{kk}.THoG, 2))) ];
end;

%----------------------------------------------------------

if does_train_CRF
	disp('loading CRF''s parameters ...');
    %save('paramsNodCRF.mat', 'paramsNodCRF', '-mat')
    load paramsNodCRF;

    %paramsNodCRF.caption = 'CRF';
    %paramsNodCRF.regFactor = 10;  % for backward compatibility. = regFactorL2.
    %paramsNodCRF.regFactorL2 = 10;  % new.
    %paramsNodCRF.regFactorL1 = 10;  % new.
    %paramsNodCRF.regL1FeatureTypes = ???;  % new: ALL, NODE, EDGE.
    %paramsNodCRF.optimizer = 'bfgs';
    %paramsNodCRF.windowSize = 0;
    %paramsNodCRF.modelType = 'crf';
    %paramsNodCRF.maxIterations = 300;
    %paramsNodCRF.debugLevel = 0;
    %paramsNodCRF.rangeWeights = [0, 0];
    %paramsNodCRF.normalizeWeights = 1;
    %paramsNodCRF.weightsInitType = ???;  % new: ZERO, CONSTANT, RANDOM, MEAN, RANDOM_MEAN_STDDEV, RANDOM_GAUSSIAN, RANDOM_GAUSSIAN2, GAUSSIAN, PREDEFINED.
    %paramsNodCRF.initWeights = ???;  % new.
end;
if does_train_HCRF
	disp('loading HCRF''s parameters ...');
    %save('paramsNodHCRF.mat', 'paramsNodHCRF', '-mat')
    load paramsNodHCRF;

    %paramsNodHCRF.caption = 'HCRF';
    %paramsNodHCRF.regFactor = 10;  % for backward compatibility. = regFactorL2.
    %paramsNodHCRF.regFactorL2 = 10;  % new.
    %paramsNodHCRF.regFactorL1 = 10;  % new.
    %paramsNodHCRF.regL1FeatureTypes = ???;  % new: ALL, NODE, EDGE.
    %paramsNodHCRF.nbHiddenStates = 3;
    %paramsNodHCRF.optimizer = 'bfgs';
    %paramsNodHCRF.windowSize = 0;
    %paramsNodHCRF.modelType = 'hcrf';
    %paramsNodHCRF.maxIterations = 300;
    %paramsNodHCRF.debugLevel = 0;
    %paramsNodHCRF.rangeWeights = [-1, 1];
    %paramsNodHCRF.normalizeWeights = 1;
    %paramsNodHCRF.windowsRecSize = 32;
    %paramsNodHCRF.initWeights = ???;  % new.
end;
if does_train_LDCRF
	disp('loading LDCRF''s parameters ...');
    %save('paramsNodLDCRF.mat', 'paramsNodLDCRF', '-mat')
    load paramsNodLDCRF;

    %paramsNodLDCRF.caption = 'LDCRF';
    %paramsNodLDCRF.regFactor = 10;  % for backward compatibility. = regFactorL2.
    %paramsNodLDCRF.regFactorL2 = 10;  % new.
    %paramsNodLDCRF.regFactorL1 = 10;  % new.
    %paramsNodLDCRF.regL1FeatureTypes = ???;  % new: ALL, NODE, EDGE.
    %paramsNodLDCRF.nbHiddenStates = 3;
    %paramsNodLDCRF.optimizer = 'bfgs';
    %paramsNodLDCRF.windowSize = 0;
    %paramsNodLDCRF.modelType = 'ldcrf';
    %paramsNodLDCRF.maxIterations = 300;
    %paramsNodLDCRF.debugLevel = 0;
    %paramsNodLDCRF.rangeWeights = [0, 0];
    %paramsNodLDCRF.normalizeWeights = 1;
    %paramsNodLDCRF.weightsInitType = ???;  % new: ZERO, CONSTANT, RANDOM, MEAN, RANDOM_MEAN_STDDEV, RANDOM_GAUSSIAN, RANDOM_GAUSSIAN2, GAUSSIAN, PREDEFINED.
    %paramsNodLDCRF.initWeights = ???;  % new.
end;

paramsData.weightsPerSequence = ones(1,128) ;
paramsData.factorSeqWeights = 1;

truth_labels = segmentation_info.truth_labels;
predicted_labels_CRF = cell(length(truth_labels), 1);
predicted_labels_HCRF = cell(length(truth_labels), 1);
predicted_labels_LDCRF = cell(length(truth_labels), 1);
predicted_gesture_labels = zeros(num_gesture_files, 1);

result_idx = 0;
R = [];
score  = [];
local_scores = [];

for hh = 1:length(num_hidden_states_list)
for ww = 1:length(window_size_list)
	resultant_file_path = strcat(resultant_file_path_prefix, sprintf('h%d_w%d_', num_hidden_states_list(hh), window_size_list(ww)), start_timestamp, '.mat');

    confusion_matrix_CRF = zeros(unique_gesture_count + 1, unique_gesture_count + 1);
    confusion_matrix_HCRF = zeros(unique_gesture_count + 1, unique_gesture_count + 1);
    confusion_matrix_LDCRF = zeros(unique_gesture_count + 1, unique_gesture_count + 1);

	%----------------------------------------------------------
	if does_train_CRF
		disp('training & testing CRF ...');

		result_idx = result_idx + 1;

	    paramsNodCRF.caption = sprintf('CRF-w%d', window_size_list(ww));
	    paramsNodCRF.regFactorL2 = regularization_L2;
	    paramsNodCRF.regFactorL1 = regularization_L1;
	    paramsNodCRF.regL1FeatureTypes = regularization_L1_feature_types;
	    paramsNodCRF.optimizer = optimizer;
	    paramsNodCRF.windowSize = window_size_list(ww);
	    paramsNodCRF.maxIterations = max_iterations;
		paramsNodCRF.normalizeWeights = 1;
		R{result_idx}.params = paramsNodCRF;

		tic;
		[R{result_idx}.model R{result_idx}.stats] = train(trainCompleteSeqs, trainCompleteLabels, R{result_idx}.params);
		fprintf(1, 'elapsed time: %f (training)', toc)
		tic;
		[R{result_idx}.ll R{result_idx}.labels] = test(R{result_idx}.model, allCompleteSeqs, allCompleteLabels);
		fprintf(1, ', %f (testing)\n', toc)

		%matLabels = cell2mat(R{result_idx}.labels);
		%matLikelihoods = cell2mat(R{result_idx}.ll);
		%[R{result_idx}.d R{result_idx}.f] = CreateROC(matLabels, matLikelihoods(2,:), R{result_idx}.params.rocRange);
		%target_class_label = 1;
		%[R{result_idx}.d R{result_idx}.f] = CreateROC_sangwook(matLabels, matLikelihoods(2,:), R{result_idx}.params.rocRange, target_class_label);

		for kk = 1:length(R{result_idx}.ll)
			confusion_matrix_CRF = compute_confusion_matrix(confusion_matrix_CRF, R{result_idx}.ll{kk}, allCompleteLabels{kk} + 1, unique_gesture_count);
		end;

		for kk = 1:length(R{result_idx}.ll)
			[max_LL_val, max_LL_idx] = max(R{result_idx}.ll{kk});
			[max_freq, max_freq_idx] = data_max_freq(max_LL_idx);

			% gesture ID: 0 (background), 1, ..., unique_gesture_count.
			predicted_gesture_labels(kk) = max_freq_idx - 1;
		end;

		idx = 1;
		for kk = 1:length(truth_labels)
			labels = zeros(length(truth_labels{kk}), 1);
			for ii = 1:length(truth_labels{kk})
				labels(ii) = predicted_gesture_labels(idx);

				idx = idx + 1;
			end;

			predicted_labels_CRF{kk} = labels;
		end;

		[score(result_idx), local_scores{result_idx}] = lscore(truth_labels, predicted_labels_CRF);

	    if ~isempty(resultant_file_path)
			save(resultant_file_path, 'R', 'confusion_matrix_CRF', 'confusion_matrix_HCRF', 'confusion_matrix_LDCRF', 'truth_labels', 'predicted_labels_CRF', 'predicted_labels_HCRF', 'predicted_labels_LDCRF', 'score', 'local_scores');
		end;
	end;

	%----------------------------------------------------------
	if does_train_HCRF
		disp('training & testing HCRF ...');

		result_idx = result_idx + 1;

	    paramsNodHCRF.caption = sprintf('HCRF-h%d-w%d', num_hidden_states_list(hh), window_size_list(ww));
	    paramsNodHCRF.regFactorL2 = regularization_L2;
	    paramsNodHCRF.regFactorL1 = regularization_L1;
	    paramsNodHCRF.regL1FeatureTypes = regularization_L1_feature_types;
	    paramsNodHCRF.nbHiddenStates = num_hidden_states_list(hh);
	    paramsNodHCRF.optimizer = optimizer;
	    paramsNodHCRF.windowSize = window_size_list(ww);
	    paramsNodHCRF.maxIterations = max_iterations;
		paramsNodHCRF.normalizeWeights = 1;
		R{result_idx}.params = paramsNodHCRF;

		tic;
		[R{result_idx}.model R{result_idx}.stats] = train(trainCompleteSeqs, trainCompleteLabels, R{result_idx}.params);
		fprintf(1, 'elapsed time: %f (training)', toc)
		tic;
		[R{result_idx}.ll R{result_idx}.labels] = test(R{result_idx}.model, allCompleteSeqs, allCompleteLabels);
		fprintf(1, ', %f (testing)\n', toc)

		%matLabels = cell2mat(R{result_idx}.labels);
		%matLikelihoods = cell2mat(R{result_idx}.ll);
		%[R{result_idx}.d R{result_idx}.f] = CreateROC(matLabels, matLikelihoods(2,:), R{result_idx}.params.rocRange);
		%target_class_label = 1;
		%[R{result_idx}.d R{result_idx}.f] = CreateROC_sangwook(matLabels, matLikelihoods(2,:), R{result_idx}.params.rocRange, target_class_label);

		for kk = 1:length(R{result_idx}.ll)
			confusion_matrix_HCRF = compute_confusion_matrix(confusion_matrix_HCRF, R{result_idx}.ll{kk}, allCompleteLabels{kk} + 1, unique_gesture_count);
		end;

		for kk = 1:length(R{result_idx}.ll)
			[max_LL_val, max_LL_idx] = max(R{result_idx}.ll{kk});
			[max_freq, max_freq_idx] = data_max_freq(max_LL_idx);

			% gesture ID: 0 (background), 1, ..., unique_gesture_count.
			predicted_gesture_labels(kk) = max_freq_idx - 1;
		end;

		idx = 1;
		for kk = 1:length(truth_labels)
			labels = zeros(length(truth_labels{kk}), 1);
			for ii = 1:length(truth_labels{kk})
				labels(ii) = predicted_gesture_labels(idx);

				idx = idx + 1;
			end;

			predicted_labels_HCRF{kk} = labels;
		end;

		[score(result_idx), local_scores{result_idx}] = lscore(truth_labels, predicted_labels_HCRF);

	    if ~isempty(resultant_file_path)
			save(resultant_file_path, 'R', 'confusion_matrix_CRF', 'confusion_matrix_HCRF', 'confusion_matrix_LDCRF', 'truth_labels', 'predicted_labels_CRF', 'predicted_labels_HCRF', 'predicted_labels_LDCRF', 'score', 'local_scores');
		end;
	end;

	%----------------------------------------------------------
	if does_train_LDCRF
		disp('training & testing LDCRF ...');

		result_idx = result_idx + 1;

	    paramsNodLDCRF.caption = sprintf('LDCRF-h%d-w%d', num_hidden_states_list(hh), window_size_list(ww));
	    paramsNodLDCRF.regFactorL2 = regularization_L2;
	    paramsNodLDCRF.regFactorL1 = regularization_L1;
	    %paramsNodLDCRF.regL1FeatureTypes = regularization_L1_feature_types;
	    paramsNodLDCRF.nbHiddenStates = num_hidden_states_list(hh);
	    paramsNodLDCRF.optimizer = optimizer;
	    paramsNodLDCRF.windowSize = window_size_list(ww);
	    paramsNodLDCRF.maxIterations = max_iterations;
		paramsNodLDCRF.normalizeWeights = 1;
		R{result_idx}.params = paramsNodLDCRF;

		tic;
		[R{result_idx}.model R{result_idx}.stats] = train(trainCompleteSeqs, trainCompleteLabels, R{result_idx}.params);
		fprintf(1, 'elapsed time: %f (training)', toc)
		tic;
		[R{result_idx}.ll R{result_idx}.labels] = test(R{result_idx}.model, allCompleteSeqs, allCompleteLabels);
		fprintf(1, ', %f (testing)\n', toc)

		%matLabels = cell2mat(R{result_idx}.labels);
		%matLikelihoods = cell2mat(R{result_idx}.ll);
		%[R{result_idx}.d R{result_idx}.f] = CreateROC(matLabels, matLikelihoods(2,:), R{result_idx}.params.rocRange);
		%target_class_label = 1;
		%[R{result_idx}.d R{result_idx}.f] = CreateROC_sangwook(matLabels, matLikelihoods(2,:), R{result_idx}.params.rocRange, target_class_label);

		for kk = 1:length(R{result_idx}.ll)
			confusion_matrix_LDCRF = compute_confusion_matrix(confusion_matrix_LDCRF, R{result_idx}.ll{kk}, allCompleteLabels{kk} + 1, unique_gesture_count);
		end;

		for kk = 1:length(R{result_idx}.ll)
			[max_LL_val, max_LL_idx] = max(R{result_idx}.ll{kk});
			[max_freq, max_freq_idx] = data_max_freq(max_LL_idx);

			% gesture ID: 0 (background), 1, ..., unique_gesture_count.
			predicted_gesture_labels(kk) = max_freq_idx - 1;
		end;

		idx = 1;
		for kk = 1:length(truth_labels)
			labels = zeros(length(truth_labels{kk}), 1);
			for ii = 1:length(truth_labels{kk})
				labels(ii) = predicted_gesture_labels(idx);

				idx = idx + 1;
			end;

			predicted_labels_LDCRF{kk} = labels;
		end;

		[score(result_idx), local_scores{result_idx}] = lscore(truth_labels, predicted_labels_LDCRF);

	    if ~isempty(resultant_file_path)
			save(resultant_file_path, 'R', 'confusion_matrix_CRF', 'confusion_matrix_HCRF', 'confusion_matrix_LDCRF', 'truth_labels', 'predicted_labels_CRF', 'predicted_labels_HCRF', 'predicted_labels_LDCRF', 'score', 'local_scores');
		end;
	end;
end;
end;
