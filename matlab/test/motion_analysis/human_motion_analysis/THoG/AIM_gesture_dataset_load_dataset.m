function [ seqs ] = AIM_gesture_dataset_load_dataset(dataset_base_directory_path, feature_directory_name, feature_file_list_file_name, feature_ext_name)

feature_file_list_file_path = strcat(dataset_base_directory_path, feature_directory_name, '\', feature_file_list_file_name);

fid = fopen(feature_file_list_file_path);
ii = 1;
while 0 == feof(fid)
	line = fgets(fid);
	if 1 == isempty(line)
		break;
	end;
	featureFileList{1, ii} = line;
	ii = ii + 1;
end;

num_dataset = length(featureFileList);

%----------------------------------------------------------
disp('reading AIM gesture dataset ...');

seqs = cell(1, num_dataset);

for ii = 1:num_dataset
	dataset_file_path = strcat(dataset_base_directory_path, feature_directory_name, '\', featureFileList{ii}, feature_ext_name);
	seqs{ii} = dlmread(dataset_file_path, ' ')';

	% remove the last row in dataset
	seqs{ii} = seqs{ii}(1:end-1,:);
end;
