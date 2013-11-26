%----------------------------------------------------------

% at desire.kaist.ac.kr
dataset_base_directory_path = 'E:\dataset\motion\ChaLearn_Gesture_Challenge_dataset\';

% at eden.kaist.ac.kr
%dataset_base_directory_path = 'F:\sangwook\dataset\motion\ChaLearn_Gesture_Challenge_dataset\';

% at WD external HDD
%dataset_base_directory_path = 'F:\dataset\motion\ChaLearn_Gesture_Challenge_dataset\';

%----------------------------------------------------------

feature_directory_path_prefix = 'quasi_lossless_format\train_data\';
feature_directory_prefix = 'devel';
feature_source_directory_suffix = '_thog3';
feature_target_directory_suffix = '_thog3_1deg_hcrf';
%feature_target_directory_suffix = '_thog3_10deg_hcrf';

segmentation_annotation_file_name = 'data_annotation\tempo_segment.csv';

feature_file_prefix = 'M_';
%feature_file_prefix = 'K_';

%feature_ext_name = 'THoG';
feature_ext_name = 'HoG';
label_ext_name = 'lbl';
seqlabel_ext_name = 'seqlbl';
segmentation_label_ext_name = 'seg_lbl';
segmentation_seqlabel_ext_name = 'seg_seqlbl';

dataset_directory_indexes = 1:20;  % devel01 ~ devel20.
feature_file_indexes = 1:47;  % M_1 ~ M_47 or K_1 ~ K_47.

%----------------------------------------------------------
disp('reading segmentation annotation info for ChaLearn Gesture Challenge dataset ...');

segmentation_annotation_file_path = strcat(dataset_base_directory_path, segmentation_annotation_file_name);
annotations = chalearn_extract_segmentation_annotation(segmentation_annotation_file_path);

%----------------------------------------------------------
disp('reading ChaLearn Gesture Challenge dataset ...');

labels = cell(length(dataset_directory_indexes), length(feature_file_indexes));
seg_labels = cell(length(dataset_directory_indexes), length(feature_file_indexes));
for dataset_id = dataset_directory_indexes
	disp(sprintf('processing dataset %d ...', dataset_id));

	dataset_id_str = sprintf('%02d', dataset_id);
	feature_source_directory_path = strcat(dataset_base_directory_path, feature_directory_path_prefix, feature_directory_prefix, dataset_id_str, feature_source_directory_suffix);

	for file_id = feature_file_indexes
		file_id_str = sprintf('%d', file_id);

		feature_file_path = strcat(feature_source_directory_path, '\', feature_file_prefix, file_id_str, '.',  feature_ext_name);
		seq = dlmread(feature_file_path, ' ');

		% remove the last row in dataset
		seq = seq(:, 1:end-1)';

		if size(seq, 1) ~= 360
			error('the length of each HoG must be 360.');
		end;

		labels{dataset_id, file_id} = zeros(1, size(seq, 2));
		seg_labels{dataset_id, file_id} = zeros(1, size(seq, 2));
	end;
end;

annotation_len = size(annotations, 1);
for ii = 1:annotation_len
	idx = annotations(ii,1);
	dataset_id = find(dataset_directory_indexes == idx);
	if length(dataset_id(:)) == 0
		continue;
	elseif length(dataset_id(:)) > 1
		error('dataset_directory_indexes has duplicate index.');
	end;

	frame_num = length(labels{dataset_id, annotations(ii,2)});
	start_frame = annotations(ii,5);
	if start_frame > frame_num
		start_frame = frame_num;
	end;
	end_frame = annotations(ii,6);
	if end_frame > frame_num
		end_frame = frame_num;
	end;
	labels{dataset_id, annotations(ii,2)}(1,start_frame:end_frame) = annotations(ii,4);
	seg_labels{dataset_id, annotations(ii,2)}(1,start_frame:end_frame) = 1;
end;

%----------------------------------------------------------
disp('writing label files for ChaLearn Gesture Challenge dataset ...');

for dataset_id = dataset_directory_indexes
	disp(sprintf('processing dataset %d ...', dataset_id));

	dataset_id_str = sprintf('%02d', dataset_id);
	feature_target_directory_path = strcat(dataset_base_directory_path, feature_directory_path_prefix, feature_directory_prefix, dataset_id_str, feature_target_directory_suffix);

	for file_id = feature_file_indexes
		file_id_str = sprintf('%d', file_id);

		label = labels{dataset_id, file_id};
		seq_label = seg_labels{dataset_id, file_id};

		% for labeling.

		% write label files for HCRF library.
		target_label_file_path = strcat(feature_target_directory_path, '\', feature_file_prefix, file_id_str, '.', label_ext_name);
		dlmwrite(target_label_file_path, size(label));
		dlmwrite(target_label_file_path, label, '-append');

		% FIXME [fix] >>
		%	Sequence label has to be only a single value for each sequence.
		%	But now there are more than one labels for test sequences.
		% write sequence label files for HCRF library.
		target_seqlabel_file_path = strcat(feature_target_directory_path, '\', feature_file_prefix, file_id_str, '.', seqlabel_ext_name);
		dlmwrite(target_seqlabel_file_path, unique(label)');

		% for segmentation.
		%	the positive labels (1) represent gesture/motion.
		%	the zero labels (0) represent transition.

		% write segmentation label files for HCRF library.
		target_label_file_path = strcat(feature_target_directory_path, '\', feature_file_prefix, file_id_str, '.', segmentation_label_ext_name);
		dlmwrite(target_label_file_path, size(seq_label));
		dlmwrite(target_label_file_path, seq_label, '-append');

		% FIXME [fix] >>
		%	Sequence label has to be only a single value for each sequence.
		%	But now there are more than one labels for test sequences.
		% write segmentation sequence label files for HCRF library.
		target_seqlabel_file_path = strcat(feature_target_directory_path, '\', feature_file_prefix, file_id_str, '.', segmentation_seqlabel_ext_name);
		dlmwrite(target_seqlabel_file_path, unique(seq_label)');
	end;
end;
