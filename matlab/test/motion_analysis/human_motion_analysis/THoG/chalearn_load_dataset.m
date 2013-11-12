function [ trainSeqs trainLabels testSeqs testLabels ] = load_chalearn_data(dataset_base_directory_path, dataset_directory_name, feature_directory_name, dataset_base_file_name, feature_ext_name)

dataset_directory_path = strcat(dataset_base_directory_path, dataset_directory_name, '\');
feature_directory_path = strcat(dataset_base_directory_path, feature_directory_name, '\');

train_label_file_path = strcat(dataset_directory_path, dataset_directory_name, '_train.csv');
test_label_file_path = strcat(dataset_directory_path, dataset_directory_name, '_test.csv');

%----------------------------------------------------------
disp('reading train & test labels of ChaLearn Gesture Challenge dataset ...');

%trainLabels = cell(1, num_train_dataset);
%testLabels = cell(1, num_test_dataset);

fid = fopen(train_label_file_path);
ii = 1;
while 0 == feof(fid)
	line = fgets(fid);
	if 1 == isempty(line)
		break;
	end;
	pieces = regexp(line, ',', 'split');
	indexes = str2num(pieces{2});
	trainLabels{ii} = indexes;
	ii = ii + 1;
end;

fid = fopen(test_label_file_path);
ii = 1;
while 0 == feof(fid)
	line = fgets(fid);
	if 1 == isempty(line)
		break;
	end;
	pieces = regexp(line, ',', 'split');
	indexes = str2num(pieces{2});
	testLabels{ii} = indexes;
	ii = ii + 1;
end;

num_dataset = 47;
num_train_dataset = length(trainLabels);
num_test_dataset = length(testLabels);
if num_test_dataset ~= num_dataset - num_train_dataset
	error('the number of test dataset is not correct ...');
end;

%----------------------------------------------------------
disp('reading ChaLearn Gesture Challenge dataset for training ...');

trainSeqs = cell(1, num_train_dataset);

for ii=1:num_train_dataset
	data_idx = sprintf('%d', ii);
	dataset_file_path = strcat(feature_directory_path, dataset_base_file_name, data_idx, feature_ext_name);
	trainSeqs{ii} = dlmread(dataset_file_path, ' ')';

	% remove the last row in train dataset
	trainSeqs{ii} = trainSeqs{ii}(1:end-1,:);
end;

%----------------------------------------------------------
disp('reading ChaLearn Gesture Challenge dataset for testing ...');

testSeqs = cell(1, num_test_dataset);

for ii=1:num_test_dataset
	data_idx = sprintf('%d', num_train_dataset + ii);
	dataset_file_path = strcat(feature_directory_path, dataset_base_file_name, data_idx, feature_ext_name);
	testSeqs{ii} = dlmread(dataset_file_path, ' ')';

	% remove the last row in train dataset
	testSeqs{ii} = testSeqs{ii}(1:end-1,:);
end;
