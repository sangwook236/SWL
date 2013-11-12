%----------------------------------------------------------

% at desire.kaist.ac.kr
%cd('D:\working_copy\research_https\matlab\human_motion_analysis\THoG');

% at eden.kaist.ac.kr
%cd('F:\sangwook\working_copy\research_https\matlab\human_motion_analysis\THoG');

%----------------------------------------------------------

% at desire.kaist.ac.kr
dataset_base_directory_path = 'E:\dataset\motion\ChaLearn_Gesture_Challenge_dataset\quasi_lossless_format\train_data\';

% at eden.kaist.ac.kr
%dataset_base_directory_path = 'F:\sangwook\dataset\motion\ChaLearn_Gesture_Challenge_dataset\quasi_lossless_format\train_data\';

% at WD external HDD
%dataset_base_directory_path = 'F:\dataset\motion\ChaLearn_Gesture_Challenge_dataset\quasi_lossless_format\train_data\';

%----------------------------------------------------------

dataset_directory_name = 'devel01';
feature_directory_name = 'devel01_thog';

%feature_type_name = 'THoG';
feature_type_name = 'HoG';

%plot_type_of_THoG = 1;
plot_type_of_THoG = 5;

does_use_RGB_image = true;

% HoG_sequence_indexes(1) ~ HoG_sequence_indexes(2)
% if index == 0, it means the first or last sequence index.
plotting_sequence_indexes = [ 0 0 ];

%----------------------------------------------------------
disp('loading HoG or THoG dataset ...');

if does_use_RGB_image
    % load HoG or THoG dataset
	[ trainSeqs trainLabels testSeqs testLabels ] = chalearn_load_dataset(dataset_base_directory_path, dataset_directory_name, feature_directory_name, 'M_', strcat('.', feature_type_name));
else
    % load HoG or THoG dataset
	[ trainSeqs trainLabels testSeqs testLabels ] = chalearn_load_dataset(dataset_base_directory_path, dataset_directory_name, feature_directory_name, 'K_', strcat('.', feature_type_name));
end;

HoG_sequences = [ trainSeqs testSeqs ];
num_HoG_train_sequences = length(trainSeqs);

clear trainSeqs trainLabels testSeqs testLabels;

%----------------------------------------------------------
% plot a surface of MovM of each sequence

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

	if ii <= num_HoG_train_sequences
        title(sprintf('seq: %d (train)', ii));
    else
        title(sprintf('seq: %d (test)', ii));
    end;
	xlabel('frame');
	ylabel('angle [deg]');
	view([0 0 1]);

    key = input('press any key to continue except for ''q'':', 's')
    if 'q' == key
        return;
    end

    clf;
end;
