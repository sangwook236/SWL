%----------------------------------------------------------

% at desire.kaist.ac.kr
%addpath('D:\working_copy\swl_https\matlab\src\statistical_analysis\directional_statistics');
%cd('D:\working_copy\research_https\matlab\human_motion_analysis\THoG');

% at eden.kaist.ac.kr
%addpath('E:\sangwook\working_copy\swl_https\matlab\src\statistical_analysis\directional_statistics');
%cd('E:\sangwook\working_copy\research_https\matlab\human_motion_analysis\THoG');

%----------------------------------------------------------

% at desire.kaist.ac.kr
%dataset_base_directory_path = 'F:\AIM_gesture_dataset\';

% at eden.kaist.ac.kr
dataset_base_directory_path = 'E:\sangwook\AIM_gesture_dataset\';

% at WD external HDD
%dataset_base_directory_path = 'F:\AIM_gesture_dataset\';

%----------------------------------------------------------

feature_directory_name = 's01_sangwook_lee_20120719_per_gesture_avi_640x480_30fps_3000kbps_thog_1deg';
feature_file_list_file_name = 'file_list_s01_thog.txt';

%feature_type_name = 'THoG';
feature_type_name = 'HoG';

HoG_bin_width = 1;  % 1 deg.
HoG_scale_factor = 2;
plot_type_of_HoG = 1;

% plotting_sequence_indexes(1):plotting_frame_indexes(1) ~ plotting_sequence_indexes(2):plotting_frame_indexes(2)
% if index == 0, it means the first or last sequence/frame index.
plotting_sequence_indexes = [ 0 0 ];
plotting_frame_indexes = [ 0 0 ];

%----------------------------------------------------------
% load HoG or THoG dataset
[ seqs ] = AIM_gesture_dataset_load_dataset(dataset_base_directory_path, feature_directory_name, feature_file_list_file_name, strcat('.', feature_type_name));

HoG_sequences = [ seqs ];
clear seqs;

%----------------------------------------------------------
% plot HoG of each frame

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
    numFrames = size(HoG_sequences{ii}, 2);

	if start_seq_idx == ii && plotting_frame_indexes(1) > 0
	    start_frame_idx = plotting_frame_indexes(1);
	else
		start_frame_idx = 1;
	end;
	if end_seq_idx == ii && plotting_frame_indexes(2) > 0
    	end_frame_idx = plotting_frame_indexes(2);
	else
		end_frame_idx = numFrames;
	end;

    if start_frame_idx > end_frame_idx || start_frame_idx < 1 || start_frame_idx > numFrames || end_frame_idx < 1 || end_frame_idx > numFrames
        error(sprintf('[SWL] start and/or end indexes of frame are incorrect - start: %d, end: %d, #frames: %d)', start_frame_idx, end_frame_idx, numFrames));
    end;

    for jj = start_frame_idx:end_frame_idx
        plot_HoG(HoG_sequences{ii}(:,jj), HoG_bin_width, HoG_scale_factor, plot_type_of_HoG);

        title(sprintf('seq: %d, time-slice: %d', ii, jj));
        legend('HoG hist', 'HoG KDE', 'Location', 'SouthOutside', 'Orientation', 'horizontal');

        key = input('press any key to continue except for ''q'':', 's')
        if 'q' == key
            return;
        end

        clf;
    end;
end;
