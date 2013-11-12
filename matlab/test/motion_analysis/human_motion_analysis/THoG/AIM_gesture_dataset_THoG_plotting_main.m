%----------------------------------------------------------

% at desire.kaist.ac.kr
%cd('D:\working_copy\research_https\matlab\human_motion_analysis\THoG');

% at eden.kaist.ac.kr
%cd('E:\sangwook\working_copy\research_https\matlab\human_motion_analysis\THoG');

%----------------------------------------------------------

% at desire.kaist.ac.kr
dataset_base_directory_path = 'F:\AIM_gesture_dataset\';

% at eden.kaist.ac.kr
%dataset_base_directory_path = 'E:\sangwook\AIM_gesture_dataset\';

% at WD external HDD
%dataset_base_directory_path = 'F:\AIM_gesture_dataset\';

%----------------------------------------------------------

feature_directory_name = 's01_sangwook_lee_20120719_per_gesture_avi_640x480_30fps_3000kbps_thog2';
feature_file_list_file_name = 'file_list_s01_thog.txt';

%feature_type_name = 'THoG';
feature_type_name = 'HoG';

%plot_type_of_THoG = 1;
plot_type_of_THoG = 5;

% HoG_sequence_indexes(1) ~ HoG_sequence_indexes(2)
% if index == 0, it means the first or last sequence index.
plotting_sequence_indexes = [ 0 0 ];

%----------------------------------------------------------
disp('loading HoG or THoG dataset ...');

% load HoG or THoG dataset
[ seqs ] = AIM_gesture_dataset_load_dataset(dataset_base_directory_path, feature_directory_name, feature_file_list_file_name, strcat('.', feature_type_name));

HoG_sequences = seqs;
clear seqs;

%----------------------------------------------------------
% plot a surface of THoG of each sequence

figure;

numSeqs = length(HoG_sequences);
if plotting_sequence_indexes(1) > 0
    start_seq_idx = plotting_sequence_indexes(1);
else
	start_seq_idx = 1;
end;
if plotting_sequence_indexes(2) > 0
    end_seq_idx = plotting_sequence_indexes(2);
else
	end_seq_idx = numSeqs;
end;

if start_seq_idx > end_seq_idx || start_seq_idx < 1 || start_seq_idx > numSeqs || end_seq_idx < 1 || end_seq_idx > numSeqs
    error(sprintf('[SWL] start and/or end indexes of sequence are incorrect - start: %d, end: %d, #sequences: %d)', start_seq_idx, end_seq_idx, numSeqs));
end;

for ii = start_seq_idx:end_seq_idx
	plot_THoG(HoG_sequences{ii}, plot_type_of_THoG);

    title(sprintf('seq: %d', ii));
	xlabel('frame');
	ylabel('angle [deg]');
	view([0 0 1]);

    key = input('press any key to continue except for ''q'':', 's')
    if 'q' == key
        return;
    end

    clf;
end;
