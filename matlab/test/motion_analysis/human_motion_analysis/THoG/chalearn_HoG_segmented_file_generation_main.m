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

segmentation_annotation_file_name = 'data_annotation\tempo_segment.csv';

feature_file_prefix = 'M_';
%feature_file_prefix = 'K_';

%feature_type_name = 'THoG';
feature_type_name = 'HoG';

degree = '1deg';
%degree = '10deg';

dataset_directory_indexes = 1:20;  % devel01 ~ devel20.

%----------------------------------------------------------
disp('reading segmentation annotation info for ChaLearn Gesture Challenge dataset ...');

segmentation_annotation_file_path = strcat(dataset_base_directory_path, segmentation_annotation_file_name);
annotations = chalearn_extract_segmentation_annotation(segmentation_annotation_file_path);

%----------------------------------------------------------
disp('writing segmented feature files for ChaLearn Gesture Challenge dataset ...');

annotation_len = size(annotations, 1);
old_dataset_id = -1;
for kk = 1:annotation_len
	dataset_idx = annotations(kk,1);

	if isempty(find(dataset_directory_indexes == dataset_idx))
		continue;
	end;

	file_idx = annotations(kk,2);
	gesture_idx = annotations(kk,3);
	gesture_label_id = annotations(kk,4);
	start_frame = annotations(kk,5);
	end_frame = annotations(kk,6);

	dataset_idx_str = sprintf('%02d', dataset_idx);
	file_idx_str = sprintf('%d', file_idx);
	gesture_idx_str = sprintf('%d', gesture_idx);

	feature_target_directory_path = strcat(dataset_base_directory_path, feature_directory_path_prefix, feature_directory_prefix, dataset_idx_str, feature_source_directory_suffix, '_', degree, '_segmented');
	if ~exist(feature_target_directory_path, 'dir')
		mkdir(strcat(dataset_base_directory_path, feature_directory_path_prefix), strcat(feature_directory_prefix, dataset_idx_str, feature_source_directory_suffix, '_', degree, '_segmented'));
	end;

	if old_dataset_id ~= dataset_idx
		%----------------------------------------------------------
		disp(sprintf('loading HoG or THoG dataset %d ...', dataset_idx));

		[ trainSeqs trainLabels testSeqs testLabels ] = chalearn_load_dataset(strcat(dataset_base_directory_path, feature_directory_path_prefix), strcat(feature_directory_prefix, dataset_idx_str), strcat(feature_directory_prefix, dataset_idx_str, feature_source_directory_suffix), feature_file_prefix, strcat('.', feature_type_name));
	
		seqs = [ trainSeqs testSeqs ];
		switch lower(degree)
		    case '1deg'
		        HoG_sequences = seqs;
		    case '10deg'
		        HoG_sequences = cell(size(seqs));
		        for ii = 1:length(seqs)
		            HoG_sequences{ii} = zeros(36, size(seqs{ii}, 2));
		            for jj = 1:36
		                HoG_sequences{ii}(jj, :) = sum(seqs{ii}(((jj-1)*10+1):(jj*10), :), 1);
		            end;
		        end;
		    otherwise
		        disp('Unknown degree.')
		end
		clear seqs;
		clear trainSeqs trainLabels testSeqs testLabels;

		old_dataset_id = dataset_idx;
	end;

	%----------------------------------------------------------
	disp(sprintf('processing dataset %d, video file %d ...', dataset_idx, file_idx));

	frame_num = size(HoG_sequences{file_idx}, 2);
	if start_frame > frame_num
		start_frame = frame_num;
	end;
	if end_frame > frame_num
		end_frame = frame_num;
	end;

	feature_seq = HoG_sequences{file_idx}(:,start_frame:end_frame);

	% write feature files.
	target_feature_file_path = strcat(feature_target_directory_path, '\', feature_file_prefix, file_idx_str, '_', gesture_idx_str, '.', feature_type_name);
	dlmwrite(target_feature_file_path, gesture_label_id);
	dlmwrite(target_feature_file_path, size(feature_seq), '-append');
	dlmwrite(target_feature_file_path, feature_seq, '-append');
end;
