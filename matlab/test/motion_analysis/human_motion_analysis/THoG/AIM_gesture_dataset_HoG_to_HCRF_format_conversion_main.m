%----------------------------------------------------------

% at desire.kaist.ac.kr
dataset_base_directory_path = 'F:\AIM_gesture_dataset_segmented\';

% at eden.kaist.ac.kr
%dataset_base_directory_path = 'E:\sangwook\AIM_gesture_dataset_segmented\';

% at WD external HDD
%dataset_base_directory_path = 'F:\AIM_gesture_dataset_segmented\';

%----------------------------------------------------------

feature_source_directory_name = 's01_sangwook_lee_20120719_per_gesture_mp4_640x480_thog';
feature_1deg_target_directory_name = 's01_sangwook_lee_20120719_per_gesture_mp4_640x480_thog_1deg_hcrf';
feature_10deg_target_directory_name = 's01_sangwook_lee_20120719_per_gesture_mp4_640x480_thog_10deg_hcrf';
feature_file_list_file_name = 'file_list_s01_thog.txt';
label_list_file_name = 'file_list_s01_thog_label.txt';

%feature_ext_name = 'THoG';
feature_ext_name = 'HoG';
label_ext_name = 'lbl';
seqlabel_ext_name = 'seqlbl';

feature_file_list_file_path = strcat(dataset_base_directory_path, feature_source_directory_name, '\', feature_file_list_file_name);
label_file_list_file_path = strcat(dataset_base_directory_path, feature_source_directory_name, '\', label_list_file_name);

fid = fopen(feature_file_list_file_path);
ii = 1;
while 0 == feof(fid)
	line = fgets(fid);
	if 1 == isempty(line)
		break;
	end;
	feature_file_list{1, ii} = line;
	ii = ii + 1;
end;
fclose(fid);

fid = fopen(label_file_list_file_path);
labelList = fscanf(fid, '%d');
fclose(fid);

num_dataset = length(feature_file_list);
num_label = length(labelList);
if num_dataset ~= num_label
	error('the number of feature files has to be equal to the number of label');
end;

%----------------------------------------------------------
disp('reading & converting AIM gesture dataset ...');

for ii = 1:num_dataset
	source_dataset_file_path = strcat(dataset_base_directory_path, feature_source_directory_name, '\', feature_file_list{ii}, '.',  feature_ext_name);
	seq_1deg = dlmread(source_dataset_file_path, ' ');

	% remove the last row in dataset
	seq_1deg = seq_1deg(:, 1:end-1)';

	if size(seq_1deg, 1) ~= 360
		error('the length of each HoG must be 360.');
	end;

	% write data files (1 deg) for HCRF library.
	target_1deg_dataset_file_path = strcat(dataset_base_directory_path, feature_1deg_target_directory_name, '\', feature_file_list{ii}, '.', feature_ext_name);
	dlmwrite(target_1deg_dataset_file_path, size(seq_1deg));
	dlmwrite(target_1deg_dataset_file_path, seq_1deg, '-append');

	% write data files (10 deg) for HCRF library.
	seq_10deg = zeros(36, size(seq_1deg, 2));
	for jj = 1:36
		seq_10deg(jj, :) = sum(seq_1deg(((jj-1)*10+1):(jj*10), :), 1);
	end;

	target_10deg_dataset_file_path = strcat(dataset_base_directory_path, feature_10deg_target_directory_name, '\', feature_file_list{ii}, '.', feature_ext_name);
	dlmwrite(target_10deg_dataset_file_path, size(seq_10deg));
	dlmwrite(target_10deg_dataset_file_path, seq_10deg, '-append');

	label = labelList(ii) * ones(1, size(seq_1deg, 2));

	% write label files for HCRF library.
	target_1deg_label_file_path = strcat(dataset_base_directory_path, feature_1deg_target_directory_name, '\', feature_file_list{ii}, '.', label_ext_name);
	dlmwrite(target_1deg_label_file_path, size(label));
	dlmwrite(target_1deg_label_file_path, label, '-append');
	target_10deg_label_file_path = strcat(dataset_base_directory_path, feature_10deg_target_directory_name, '\', feature_file_list{ii}, '.', label_ext_name);
	dlmwrite(target_10deg_label_file_path, size(label));
	dlmwrite(target_10deg_label_file_path, label, '-append');

	% write sequence label files for HCRF library.
	target_1deg_seqlabel_file_path = strcat(dataset_base_directory_path, feature_1deg_target_directory_name, '\', feature_file_list{ii}, '.', seqlabel_ext_name);
	dlmwrite(target_1deg_seqlabel_file_path, labelList(ii));
	target_10deg_seqlabel_file_path = strcat(dataset_base_directory_path, feature_10deg_target_directory_name, '\', feature_file_list{ii}, '.', seqlabel_ext_name);
	dlmwrite(target_10deg_seqlabel_file_path, labelList(ii));
end;
