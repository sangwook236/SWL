%----------------------------------------------------------

% at desire.kaist.ac.kr
%addpath('D:\working_copy\swl_https\matlab\src\statistical_analysis\directional_statistics');
%cd('D:\working_copy\research_https\matlab\human_motion_analysis\THoG');

% at eden.kaist.ac.kr
addpath('E:\sangwook\working_copy\swl_https\matlab\src\statistical_analysis\directional_statistics');
%cd('E:\sangwook\working_copy\research_https\matlab\human_motion_analysis\THoG');

%----------------------------------------------------------
%MovM_data_file_path = './AIM_gesture_dataset_HoG_MovM/AIM_gesture_dataset_s01_sangwook_lee_1deg_4clusters_20131017/AIM_gesture_dataset_s01_sangwook_lee_HoG_MovM_integrated_20131017.mat';
%MovM_data_file_path = './AIM_gesture_dataset_HoG_MovM/AIM_gesture_dataset_s01_sangwook_lee_1deg_8clusters_20131023/AIM_gesture_dataset_s01_sangwook_lee_HoG_MovM_integrated_20131023.mat';
MovM_data_file_path = './AIM_gesture_dataset_HoG_MovM/AIM_gesture_dataset_s01_sangwook_lee_HoG_MovM_20131108T213445.mat';

%plot_type_of_MovM_map = 1;
plot_type_of_MovM_map = 5;

% plotting_sequence_indexes(1) ~ plotting_sequence_indexes(2)
% if index == 0, it means the first or last sequence index.
plotting_sequence_indexes = [ 0 0 ];

%----------------------------------------------------------
% load MovM data
MovM_sequences = load(MovM_data_file_path);

%----------------------------------------------------------
% plot a surface of MovM of each sequence

figure;

numSeqs = length(MovM_sequences.Mu);
%numSeqs2 = length(MovM_sequences.Kappa);
%numSeqs3 = length(MovM_sequences.Alpha);
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
	for jj = 1:size(MovM_sequences.Alpha{ii}, 2)
		if sum(MovM_sequences.Alpha{ii}(:,jj)) < 1e-5
			sprintf('***** MovM undefined at seq: %d, time-slice: %d', ii, jj)
		end;
	end;

	plot_MovM_map(MovM_sequences.Mu{ii}, MovM_sequences.Kappa{ii}, MovM_sequences.Alpha{ii}, plot_type_of_MovM_map);

    title(sprintf('seq: %d', ii));
	xlabel('frame');
	ylabel('angle [rad]');
	view([0 0 1]);

    key = input('press any key to continue except for ''q'':', 's')
    if 'q' == key
        return;
    end

    clf;
end;
