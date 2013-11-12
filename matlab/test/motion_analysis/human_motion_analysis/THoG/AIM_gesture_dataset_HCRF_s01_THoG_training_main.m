%----------------------------------------------------------
%addpath('D:\work\sw_dev\cpp\rnd\src\probabilistic_graphical_model\crf\hcrf\HCRF2.0b\bin\openMP');
%addpath('D:\work\sw_dev\cpp\rnd\src\probabilistic_graphical_model\crf\hcrf\HCRF2.0b\samples\matlab');

%----------------------------------------------------------
disp('reading train dataset ...');

num_train_dataset = 9;

trainSeqs = cell(1, num_train_dataset);
trainLabels = cell(1, num_train_dataset);

trainSeqs{1} = dlmread('F:\AIM_gesture_dataset\s01_sangwook_lee_20120719_per_gesture_avi_640x480_30fps_3000kbps_output\s01_g01_1_ccw_normal.THoG', ' ')';
trainSeqs{2} = dlmread('F:\AIM_gesture_dataset\s01_sangwook_lee_20120719_per_gesture_avi_640x480_30fps_3000kbps_output\s01_g01_5_cw_normal.THoG', ' ')';
trainSeqs{3} = dlmread('F:\AIM_gesture_dataset\s01_sangwook_lee_20120719_per_gesture_avi_640x480_30fps_3000kbps_output\s01_g02_1_ccw_normal.THoG', ' ')';
trainSeqs{4} = dlmread('F:\AIM_gesture_dataset\s01_sangwook_lee_20120719_per_gesture_avi_640x480_30fps_3000kbps_output\s01_g02_5_cw_normal.THoG', ' ')';
trainSeqs{5} = dlmread('F:\AIM_gesture_dataset\s01_sangwook_lee_20120719_per_gesture_avi_640x480_30fps_3000kbps_output\s01_g03_1_ccw_normal.THoG', ' ')';
trainSeqs{6} = dlmread('F:\AIM_gesture_dataset\s01_sangwook_lee_20120719_per_gesture_avi_640x480_30fps_3000kbps_output\s01_g03_5_cw_normal.THoG', ' ')';
trainSeqs{7} = dlmread('F:\AIM_gesture_dataset\s01_sangwook_lee_20120719_per_gesture_avi_640x480_30fps_3000kbps_output\s01_g04_1_normal.THoG', ' ')';
trainSeqs{8} = dlmread('F:\AIM_gesture_dataset\s01_sangwook_lee_20120719_per_gesture_avi_640x480_30fps_3000kbps_output\s01_g05_1_normal.THoG', ' ')';
trainSeqs{9} = dlmread('F:\AIM_gesture_dataset\s01_sangwook_lee_20120719_per_gesture_avi_640x480_30fps_3000kbps_output\s01_g06_1_normal.THoG', ' ')';

% remove the last row in train dataset
for ii = 1:num_train_dataset
	trainSeqs{ii} = trainSeqs{ii}(1:end-1,:);
end;

for ii = 1:num_train_dataset
	trainLabels{ii} = ones(1, size(trainSeqs{ii}, 2)) * ii;
end;

%----------------------------------------------------------
disp('reading test dataset ...');

test_method = 1;
if 1 == test_method
	num_test_dataset = 1;

	testSeqs = cell(1, num_test_dataset);
	testLabels = cell(1, num_test_dataset);

	testSeqs{1} = dlmread('F:\AIM_gesture_dataset\s01_sangwook_lee_20120719_per_gesture_avi_640x480_30fps_3000kbps_output\s01_g01_4_ccw_normal.THoG', ' ')';
else
	num_test_dataset = num_train_dataset;

	testSeqs = cell(1, num_test_dataset);
	testLabels = cell(1, num_test_dataset);

	testSeqs{1} = dlmread('F:\AIM_gesture_dataset\s01_sangwook_lee_20120719_per_gesture_avi_640x480_30fps_3000kbps_output\s01_g01_4_ccw_normal.THoG', ' ')';
	testSeqs{2} = dlmread('F:\AIM_gesture_dataset\s01_sangwook_lee_20120719_per_gesture_avi_640x480_30fps_3000kbps_output\s01_g01_8_cw_normal.THoG', ' ')';
	testSeqs{3} = dlmread('F:\AIM_gesture_dataset\s01_sangwook_lee_20120719_per_gesture_avi_640x480_30fps_3000kbps_output\s01_g02_4_ccw_normal.THoG', ' ')';
	testSeqs{4} = dlmread('F:\AIM_gesture_dataset\s01_sangwook_lee_20120719_per_gesture_avi_640x480_30fps_3000kbps_output\s01_g02_8_cw_normal.THoG', ' ')';
	testSeqs{5} = dlmread('F:\AIM_gesture_dataset\s01_sangwook_lee_20120719_per_gesture_avi_640x480_30fps_3000kbps_output\s01_g03_4_ccw_normal.THoG', ' ')';
	testSeqs{6} = dlmread('F:\AIM_gesture_dataset\s01_sangwook_lee_20120719_per_gesture_avi_640x480_30fps_3000kbps_output\s01_g03_8_cw_normal.THoG', ' ')';
	testSeqs{7} = dlmread('F:\AIM_gesture_dataset\s01_sangwook_lee_20120719_per_gesture_avi_640x480_30fps_3000kbps_output\s01_g04_3_normal.THoG', ' ')';
	testSeqs{8} = dlmread('F:\AIM_gesture_dataset\s01_sangwook_lee_20120719_per_gesture_avi_640x480_30fps_3000kbps_output\s01_g05_3_normal.THoG', ' ')';
	testSeqs{9} = dlmread('F:\AIM_gesture_dataset\s01_sangwook_lee_20120719_per_gesture_avi_640x480_30fps_3000kbps_output\s01_g06_3_normal.THoG', ' ')';
end;

% remove the last row in train dataset
for ii = 1:num_test_dataset
	testSeqs{ii} = testSeqs{ii}(1:end-1,:);
end;

for ii = 1:num_test_dataset
	testLabels{ii} = ones(1, size(testSeqs{ii}, 2)) * ii;
end;

%----------------------------------------------------------
disp('loading model data ...');

%save('paramsNodCRF.mat', 'paramsNodCRF', '-mat')
%save('paramsNodHCRF.mat', 'paramsNodHCRF', '-mat')
%save('paramsNodLDCRF.mat', 'paramsNodLDCRF', '-mat')

load paramsNodCRF;
load paramsNodHCRF;
load paramsNodLDCRF;

paramsData.weightsPerSequence = ones(1,128) ;
paramsData.factorSeqWeights = 1;

runCRF = true;
runHCRF = false;
runLDCRF = true;

result_id = 0;

%----------------------------------------------------------
if true == runCRF
	disp('processing CRF ...');

	result_id = result_id + 1;

	paramsNodCRF.normalizeWeights = 1;
	R{result_id}.params = paramsNodCRF;
	tic;
	[R{result_id}.model R{result_id}.stats] = train(trainSeqs, trainLabels, R{result_id}.params);
	fprintf(1, 'elapsed time: %f (training)', toc)
	tic;
	[R{result_id}.ll R{result_id}.labels] = test(R{result_id}.model, testSeqs, testLabels);
	fprintf(1, ', %f (testing)\n', toc)

	matLabels = cell2mat(R{result_id}.labels);
	matLikelihoods = cell2mat(R{result_id}.ll);
	[R{result_id}.d R{result_id}.f] = CreateROC_sangwook(matLabels, matLikelihoods(2,:), R{result_id}.params.rocRange);
end;

%----------------------------------------------------------
if true == runHCRF
	disp('processing HCRF ...');

	result_id = result_id + 1;

	paramsNodHCRF.normalizeWeights = 1;
	R{result_id}.params = paramsNodHCRF;
	tic;
	[R{result_id}.model R{result_id}.stats] = train(trainCompleteSeqs, trainCompleteLabels, R{result_id}.params);
	fprintf(1, 'elapsed time: %f (training)', toc)
	tic;
	[R{result_id}.ll R{result_id}.labels] = test(R{result_id}.model, testSeqs, testLabels);
	fprintf(1, ', %f (testing)\n', toc)

	matLabels = cell2mat(R{result_id}.labels);
	matLikelihoods = cell2mat(R{result_id}.ll);
	[R{result_id}.d R{result_id}.f] = CreateROC_sangwook(matLabels, matLikelihoods(2,:), R{result_id}.params.rocRange);
end;

%----------------------------------------------------------
if true == runLDCRF
	disp('processing LDCRF ...');

	result_id = result_id + 1;

	paramsNodLDCRF.normalizeWeights = 1;
	R{result_id}.params = paramsNodLDCRF;
	tic;
	[R{result_id}.model R{result_id}.stats] = train(trainSeqs, trainLabels, R{result_id}.params);
	fprintf(1, 'elapsed time: %f (training)', toc)
	tic;
	[R{result_id}.ll R{result_id}.labels] = test(R{result_id}.model, testSeqs, testLabels);
	fprintf(1, ', %f (testing)\n', toc)

	matLabels = cell2mat(R{result_id}.labels);
	matLikelihoods = cell2mat(R{result_id}.ll);
	[R{result_id}.d R{result_id}.f] = CreateROC_sangwook(matLabels, matLikelihoods(2,:), R{result_id}.params.rocRange);
end;

%----------------------------------------------------------
disp('plotting ROC ...');

plotResults(R);
