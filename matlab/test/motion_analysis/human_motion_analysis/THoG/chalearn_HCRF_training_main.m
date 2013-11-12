%----------------------------------------------------------

% at desire.kaist.ac.kr
%addpath('D:\work\sw_dev\cpp\rnd\src\probabilistic_graphical_model\crf\hcrf\HCRF2.0b\bin\openMP');
%addpath('D:\work\sw_dev\cpp\rnd\src\probabilistic_graphical_model\crf\hcrf\HCRF2.0b\samples\matlab');
%cd('D:\working_copy\research_https\matlab\human_motion_analysis\THoG');

% at eden.kaist.ac.kr
addpath('E:\sangwook\work\sw_dev\cpp\rnd\src\probabilistic_graphical_model\crf\hcrf\HCRF2.0b\bin\openMP');
addpath('E:\sangwook\work\sw_dev\cpp\rnd\src\probabilistic_graphical_model\crf\hcrf\HCRF2.0b\samples\matlab');
%cd('E:\sangwook\working_copy\research_https\matlab\human_motion_analysis\THoG');

% at LG notebook
%addpath('D:\sangwook\work\sw_dev\cpp\rnd\src\probabilistic_graphical_model\crf\hcrf\HCRF2.0b\bin\openMP');
%addpath('D:\sangwook\work\sw_dev\cpp\rnd\src\probabilistic_graphical_model\crf\hcrf\HCRF2.0b\samples\matlab');
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
%feature_directory_name = 'devel01_thog2_1deg_hcrf';
feature_directory_name = 'devel01_thog2_10deg_hcrf';

feature_file_prefix = 'M_';
%feature_file_prefix = 'K_';

%feature_type_name = 'THoG';
feature_type_name = 'HoG';

is_segmenting = false;
if is_segmenting
	% for segmentation.
	label_ext_name = 'seg_lbl';
	seqlabel_ext_name = 'seg_seqlbl';
else
	% for sequence labeling.
	label_ext_name = 'lbl';
	seqlabel_ext_name = 'seqlbl';
end;

feature_file_indexes = 1:47;  % M_1 ~ M_47 or K_1 ~ K_47.

does_train_CRF = false;
does_train_HCRF = false;  % NOTICE [caution] >> sequence labels may be incorrect. state labels are only correct.
does_train_LDCRF = true;

regularization_L2 = 10;
regularization_L1 = 10;
regularization_L1_feature_types = 'ALL';  % ALL, NODE, EDGE.
%optimizer = 'bfgs';  % cpu low, memory high.
optimizer = 'lbfgs';  % cpu high, memory low.
max_iterations = 300;

num_hidden_states_list = 7:12;
%num_hidden_states_list = 9;
%window_size_list = 11:15;
window_size_list = 8;

%----------------------------------------------------------
disp('loading HoG or THoG dataset ...');

num_feature_files = length(feature_file_indexes);

datasetList = cell(1, num_feature_files);
labelList = cell(1, num_feature_files);
seqLabelList = cell(1, num_feature_files);

ii = 1;
for file_idx = feature_file_indexes
	file_idx_str = sprintf('%d', file_idx);

	feature_file_path = strcat(dataset_base_directory_path, '\', feature_directory_name, '\', feature_file_prefix, file_idx_str, '.',  feature_type_name);
	label_file_path = strcat(dataset_base_directory_path, '\', feature_directory_name, '\', feature_file_prefix, file_idx_str, '.',  label_ext_name);
	sequence_label_file_path = strcat(dataset_base_directory_path, '\', feature_directory_name, '\', feature_file_prefix, file_idx_str, '.',  seqlabel_ext_name);

	datasetList{ii} = dlmread(feature_file_path, ',', 1, 0);
	labelList{ii} = dlmread(label_file_path, ',', 1, 0);
	seqLabelList{ii} = dlmread(sequence_label_file_path, ',');

	ii = ii + 1;
end;

train_dataset_info_file_path = strcat(dataset_base_directory_path, '\', dataset_directory_name, '\', dataset_directory_name, '_train.csv');
fid = fopen(train_dataset_info_file_path);
train_dataset_info = textscan(fid, '%s%d', 'Delimiter', ',');
fclose(fid);

num_train_dataset = length(train_dataset_info{1});

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

result_idx = 0;
R = [];

start_timestamp = datestr(clock, 30);
resultant_HCRF_file_path = strcat('chalearn_', dataset_directory_name, '_', feature_file_prefix, feature_type_name, '_HCRF_result_', start_timestamp, '.mat');

for hh = 1:length(num_hidden_states_list)
for ww = 1:length(window_size_list)
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
		[R{result_idx}.model R{result_idx}.stats] = train(datasetList(1:num_train_dataset), labelList(1:num_train_dataset), R{result_idx}.params);
		fprintf(1, 'elapsed time: %f (training)', toc)
		tic;
		[R{result_idx}.ll R{result_idx}.labels] = test(R{result_idx}.model, datasetList((num_train_dataset+1):end), labelList((num_train_dataset+1):end));
		fprintf(1, ', %f (testing)\n', toc)

		matLabels = cell2mat(R{result_idx}.labels);
		matLikelihoods = cell2mat(R{result_idx}.ll);
		[R{result_idx}.d R{result_idx}.f] = CreateROC(matLabels, matLikelihoods(2,:), R{result_idx}.params.rocRange);
		%target_class_label = 1;
		%[R{result_idx}.d R{result_idx}.f] = CreateROC_sangwook(matLabels, matLikelihoods(2,:), R{result_idx}.params.rocRange, target_class_label);
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
		%[R{result_idx}.model R{result_idx}.stats] = train(datasetList(1:num_train_dataset), seqLabelList(1:num_train_dataset), R{result_idx}.params);
		[R{result_idx}.model R{result_idx}.stats] = train(datasetList, trainLabelsPerFrame, R{result_idx}.params);
		fprintf(1, 'elapsed time: %f (training)', toc)
		tic;
		[R{result_idx}.ll R{result_idx}.labels] = test(R{result_idx}.model, datasetList((num_train_dataset+1):end), seqLabelList((num_train_dataset+1):end));
		fprintf(1, ', %f (testing)\n', toc)

		matLabels = cell2mat(R{result_idx}.labels);
		matLikelihoods = cell2mat(R{result_idx}.ll);
		[R{result_idx}.d R{result_idx}.f] = CreateROC(matLabels, matLikelihoods(2,:), R{result_idx}.params.rocRange);
		%target_class_label = 1;
		%[R{result_idx}.d R{result_idx}.f] = CreateROC_sangwook(matLabels, matLikelihoods(2,:), R{result_idx}.params.rocRange, target_class_label);
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
		[R{result_idx}.model R{result_idx}.stats] = train(datasetList(1:num_train_dataset), labelList(1:num_train_dataset), R{result_idx}.params);
		fprintf(1, 'elapsed time: %f (training)', toc)
		tic;
		[R{result_idx}.ll R{result_idx}.labels] = test(R{result_idx}.model, datasetList((num_train_dataset+1):end), labelList((num_train_dataset+1):end));
		fprintf(1, ', %f (testing)\n', toc)

		matLabels = cell2mat(R{result_idx}.labels);
		matLikelihoods = cell2mat(R{result_idx}.ll);
		[R{result_idx}.d R{result_idx}.f] = CreateROC(matLabels, matLikelihoods(2,:), R{result_idx}.params.rocRange);
		%target_class_label = 1;
		%[R{result_idx}.d R{result_idx}.f] = CreateROC_sangwook(matLabels, matLikelihoods(2,:), R{result_idx}.params.rocRange, target_class_label);
	end;

    if ~isempty(resultant_HCRF_file_path)
	    save(resultant_HCRF_file_path, 'R');
	end;
end;
end;

%----------------------------------------------------------
disp('plotting ROC ...');

if 0 == isempty(R)
	plotResults(R);
end;
