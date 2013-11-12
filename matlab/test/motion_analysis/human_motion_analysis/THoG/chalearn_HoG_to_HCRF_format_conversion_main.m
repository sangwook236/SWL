%----------------------------------------------------------

% at desire.kaist.ac.kr
dataset_base_directory_path = 'E:\dataset\motion\ChaLearn_Gesture_Challenge_dataset\quasi_lossless_format\train_data\';

% at eden.kaist.ac.kr
%dataset_base_directory_path = 'F:\sangwook\dataset\motion\ChaLearn_Gesture_Challenge_dataset\quasi_lossless_format\train_data\';

% at WD external HDD
%dataset_base_directory_path = 'F:\dataset\motion\ChaLearn_Gesture_Challenge_dataset\quasi_lossless_format\train_data\';

%----------------------------------------------------------

feature_source_directory_prefix = 'devel';
feature_source_directory_suffix = '_thog2';
feature_1deg_target_directory_suffix = '_thog2_1deg_hcrf';
feature_10deg_target_directory_suffix = '_thog2_10deg_hcrf';

feature_file_prefix = 'M_';
%feature_file_prefix = 'K_';

%feature_ext_name = 'THoG';
feature_ext_name = 'HoG';

dataset_directory_indexes = 1:20;  % devel01 ~ devel20.
feature_file_indexes = 1:47;  % M_1 ~ M_47 or K_1 ~ K_47.

%----------------------------------------------------------
disp('reading & converting ChaLearn Gesture Challenge dataset ...');

for dataset_id = dataset_directory_indexes
	disp(sprintf('processing dataset %d ...', dataset_id));

	dataset_id_str = sprintf('%02d', dataset_id);
	feature_source_directory_path = strcat(dataset_base_directory_path, feature_source_directory_prefix, dataset_id_str, feature_source_directory_suffix);
	feature_1deg_target_directory_path = strcat(dataset_base_directory_path, feature_source_directory_prefix, dataset_id_str, feature_1deg_target_directory_suffix);
	feature_10deg_target_directory_path = strcat(dataset_base_directory_path, feature_source_directory_prefix, dataset_id_str, feature_10deg_target_directory_suffix);

	for file_id = feature_file_indexes
		file_id_str = sprintf('%d', file_id);
		feature_file_name = strcat(feature_file_prefix, file_id_str, '.',  feature_ext_name);

		source_dataset_file_path = strcat(feature_source_directory_path, '\', feature_file_name);
		seq_1deg = dlmread(source_dataset_file_path, ' ');

		% remove the last row in dataset
		seq_1deg = seq_1deg(:, 1:end-1)';

		if size(seq_1deg, 1) ~= 360
			error('the length of each HoG must be 360.');
		end;

		% write data files (1 deg) for HCRF library.
		target_1deg_dataset_file_path = strcat(feature_1deg_target_directory_path, '\', feature_file_name);
		dlmwrite(target_1deg_dataset_file_path, size(seq_1deg));
		dlmwrite(target_1deg_dataset_file_path, seq_1deg, '-append');

		% write data files (10 deg) for HCRF library.
		seq_10deg = zeros(36, size(seq_1deg, 2));
		for jj = 1:36
			seq_10deg(jj, :) = sum(seq_1deg(((jj-1)*10+1):(jj*10), :), 1);
		end;

		target_10deg_dataset_file_path = strcat(feature_10deg_target_directory_path, '\', feature_file_name);
		dlmwrite(target_10deg_dataset_file_path, size(seq_10deg));
		dlmwrite(target_10deg_dataset_file_path, seq_10deg, '-append');
	end;
end;
