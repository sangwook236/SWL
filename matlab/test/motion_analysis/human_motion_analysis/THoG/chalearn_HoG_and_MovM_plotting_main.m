%----------------------------------------------------------

% at desire.kaist.ac.kr
%addpath('D:\working_copy\swl_https\matlab\src\statistical_analysis\directional_statistics');
%cd('D:\working_copy\research_https\matlab\human_motion_analysis\THoG');

% at eden.kaist.ac.kr
addpath('E:\sangwook\working_copy\swl_https\matlab\src\statistical_analysis\directional_statistics');
%cd('E:\sangwook\working_copy\research_https\matlab\human_motion_analysis\THoG');

%----------------------------------------------------------

% at desire.kaist.ac.kr
%dataset_base_directory_path = 'E:\dataset\motion\ChaLearn_Gesture_Challenge_dataset\quasi_lossless_format\train_data\';

% at eden.kaist.ac.kr
dataset_base_directory_path = 'E:\sangwook\dataset\motion\ChaLearn_Gesture_Challenge_dataset\quasi_lossless_format\train_data\';

% at WD external HDD
%dataset_base_directory_path = 'F:\dataset\motion\ChaLearn_Gesture_Challenge_dataset\quasi_lossless_format\train_data\';

%----------------------------------------------------------

dataset_directory_name = 'devel01';
feature_directory_name = 'devel01_thog2';

MovM_M_data_file_path = './chalearn_HoG_MovM/chalearn_HoG_MovM_1deg_4clusters_with_no_motion_20131027/chalearn_devel01_M_HoG_MovM_20131105T193841.mat';
MovM_K_data_file_path = './chalearn_HoG_MovM/chalearn_HoG_MovM_1deg_4clusters_with_no_motion_20131027/chalearn_devel01_M_HoG_MovM_20131105T193841.mat';

%feature_type_name = 'THoG';
feature_type_name = 'HoG';

does_use_RGB_image = true;

HoG_bin_width = 1;  % 1 deg.
HoG_scale_factor = 2;
plot_type_of_HoG_and_MovM = 1;

% plotting_sequence_indexes(1):plotting_frame_indexes(1) ~ plotting_sequence_indexes(2):plotting_frame_indexes(2)
% if index == 0, it means the first or last sequence/frame index.
plotting_sequence_indexes = [ 0 0 ];
plotting_frame_indexes = [ 0 0 ];

%----------------------------------------------------------
if true == does_use_RGB_image
    % load HoG or THoG dataset
	[ trainSeqs trainLabels testSeqs testLabels ] = chalearn_load_dataset(dataset_base_directory_path, dataset_directory_name, feature_directory_name, 'M_', strcat('.', feature_type_name));
	% load MovM data
	MovM_sequences = load(MovM_M_data_file_path);
else
    % load HoG or THoG dataset
	[ trainSeqs trainLabels testSeqs testLabels ] = chalearn_load_dataset(dataset_base_directory_path, dataset_directory_name, feature_directory_name, 'K_', strcat('.', feature_type_name));
	% load MovM data
	MovM_sequences = load(MovM_K_data_file_path);
end;

HoG_sequences = [ trainSeqs testSeqs ];
clear trainSeqs trainLabels testSeqs testLabels;

%----------------------------------------------------------
% plot HoG & MovM of each frame

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
    	if sum(MovM_sequences.Alpha{ii}(:,jj)) < 1e-5
    		sprintf('***** MovM undefined at seq. file: %d, time-slice: %d', ii, jj)
    		continue;
    	end;

        plot_HoG_and_MovM(HoG_sequences{ii}(:,jj), MovM_sequences.Mu{ii}(:,jj), MovM_sequences.Kappa{ii}(:,jj), MovM_sequences.Alpha{ii}(:,jj), HoG_bin_width, HoG_scale_factor, plot_type_of_HoG_and_MovM);

        title(sprintf('seq: %d, time-slice: %d', ii, jj));
        legend('HoG hist', 'HoG KDE', 'MovM', 'Location', 'SouthOutside', 'Orientation', 'horizontal');

        key = input('press any key to continue except for ''q'':', 's')
        if 'q' == key
            return;
        end

        clf;
    end;
end;
