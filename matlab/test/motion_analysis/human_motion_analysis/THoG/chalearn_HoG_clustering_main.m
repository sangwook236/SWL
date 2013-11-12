%----------------------------------------------------------

% at desire.kaist.ac.kr
%addpath('D:\work\sw_dev\matlab\rnd\src\machine_learning\spectral_clustering\parallel_spectral_clustering\spectralclustering-1.1');
%addpath('D:\working_copy\swl_https\matlab\src\statistical_analysis');
%cd('D:\working_copy\research_https\matlab\human_motion_analysis\THoG');

% at eden.kaist.ac.kr
%addpath('E:\sangwook\work\sw_dev\matlab\rnd\src\machine_learning\spectral_clustering\parallel_spectral_clustering\spectralclustering-1.1');
%addpath('E:\sangwook\working_copy\swl_https\matlab\src\statistical_analysis');
%cd('E:\sangwook\working_copy\research_https\matlab\human_motion_analysis\THoG');

% at LG notebook
%addpath('D:\sangwook\work\sw_dev\matlab\rnd\src\machine_learning\spectral_clustering\parallel_spectral_clustering\spectralclustering-1.1');
%addpath('D:\sangwook\working_copy\swl_https\matlab\src\statistical_analysis');
%cd('D:\sangwook\working_copy\research_https\matlab\human_motion_analysis\THoG');

%----------------------------------------------------------

% at desire.kaist.ac.kr
%dataset_base_directory_path = 'E:\dataset\motion\ChaLearn_Gesture_Challenge_dataset\quasi_lossless_format\train_data\';

% at eden.kaist.ac.kr
dataset_base_directory_path = 'E:\sangwook\dataset\motion\ChaLearn_Gesture_Challenge_dataset\quasi_lossless_format\train_data\';

% at WD external HDD
%dataset_base_directory_path = 'F:\dataset\motion\ChaLearn_Gesture_Challenge_dataset\quasi_lossless_format\train_data\';

%----------------------------------------------------------

dataset_directory_name = 'devel01';
feature_directory_name = 'devel01_thog';

%feature_type_name = 'THoG';
feature_type_name = 'HoG';

does_use_RGB_image = true;
does_generate_distance_matrix = true;
does_run_sc_algorithms = true;
does_run_nystrom_algorithms = false;
does_run_kmeans_algorithm = true;

num_neighbors = 50;
block_size = 10;
distance_matrix_save_type = 0;
%histogram_comparison_method = 'correl';
%histogram_comparison_method = 'chisqr';
%histogram_comparison_method = 'intersect';
histogram_comparison_method = 'bhatta';

num_clusters = 18;

if does_run_sc_algorithms
    sc_sigma = 20;
    sc_num_clusters = num_clusters;
end;

if does_run_nystrom_algorithms
    nystrom_num_samples = 200;
    nystrom_sigma = 20;
    nystrom_num_clusters = num_clusters;
end;

if does_run_kmeans_algorithm
    kmeans_centers = 'random';
    kmeans_num_clusters = num_clusters;
end;

%----------------------------------------------------------
disp('loading HoG or THoG dataset ...');

start_timestamp = datestr(clock, 30);
if does_use_RGB_image
	distance_matrix_base_file_name = strcat('chalearn_', dataset_directory_name, '_M_', feature_type_name, '_clustering_');

    % load HoG or THoG dataset
	[ trainSeqs trainLabels testSeqs testLabels ] = chalearn_load_dataset(dataset_base_directory_path, dataset_directory_name, feature_directory_name, 'M_', strcat('.', feature_type_name));
else
	distance_matrix_base_file_name = strcat('chalearn_', dataset_directory_name, '_K_', feature_type_name, '_clustering_');

    % load HoG or THoG dataset
	[ trainSeqs trainLabels testSeqs testLabels ] = chalearn_load_dataset(dataset_base_directory_path, dataset_directory_name, feature_directory_name, 'K_', strcat('.', feature_type_name));
end;
resultant_clustering_file_path = strcat(distance_matrix_base_file_name, histogram_comparison_method, '_', start_timestamp, '.mat');

HoG_sequences = [ trainSeqs testSeqs ];
HoG_labels = [ trainLabels testLabels ];

clear trainSeqs trainLabels testSeqs testLabels;

numSeqs = length(HoG_sequences);
dim = size(HoG_sequences{1}, 1);  % 360

numTotalHoGs = 0;
for ii = 1:numSeqs
	numTotalHoGs = numTotalHoGs + size(HoG_sequences{ii}, 2);
end;

HoG_sequences2 = zeros(dim, numTotalHoGs);
startIdx = 1;
for ii = 1:numSeqs
	endIdx = startIdx + size(HoG_sequences{ii}, 2);
	HoG_sequences2(:,startIdx:(endIdx-1)) = HoG_sequences{ii};
	startIdx = endIdx;
end;

%----------------------------------------------------------
% using parallel spectral clustering
%	[ref] http://alumni.cs.ucsb.edu/~wychen/sc.html

if does_generate_distance_matrix
	disp('generating sparse symmetric distance matrix using t-nearest-neighbor method ...');
	gen_nn_distance_sangwook(HoG_sequences2', num_neighbors, block_size, distance_matrix_save_type, histogram_comparison_method, distance_matrix_base_file_name)
end;

%----------------------------------------------------------
% using parallel spectral clustering
%	[ref] http://alumni.cs.ucsb.edu/~wychen/sc.html

if (distance_matrix_save_type == 0) || (distance_matrix_save_type == 2)
	distance_matrix_file_name = [distance_matrix_base_file_name, histogram_comparison_method, '_', num2str(num_neighbors), '_NN_sym_distance.mat'];
elseif (distance_matrix_save_type == 1) || (distance_matrix_save_type == 2)
	distance_matrix_file_name = [distance_matrix_base_file_name, histogram_comparison_method, '_', num2str(num_neighbors), '_NN_sym_distance.txt'];
end;

%----------------------------------------------------------

disp('loading distance matrix ...');
dist_mat = load(distance_matrix_file_name);

if does_run_sc_algorithms
    disp('running spectral clustering using a sparse similarity matrix ...');
    [sc_cluster_labels, evd_time1, kmeans_time1, total_time1] = sc(dist_mat.A, sc_sigma, sc_num_clusters);

    disp('saving spectral clustering results ...');
    if exist(resultant_clustering_file_path, 'file')
        save(resultant_clustering_file_path, 'sc_cluster_labels', '-append');
    else
        save(resultant_clustering_file_path, 'sc_cluster_labels');
    end;
end;

if does_run_nystrom_algorithms
    disp('running spectral clustering using Nystrom method with orthogonalization ...');
    [nystrom_cluster_labels, evd_time2, kmeans_time2, total_time2] = nystrom(HoG_sequences2', nystrom_num_samples, nystrom_sigma, nystrom_num_clusters);

    disp('running spectral clustering using Nystrom method without orthogonalization ...');
    [nystrom_no_orth_cluster_labels, evd_time3, kmeans_time3, total_time3] = nystrom_no_orth(HoG_sequences2', nystrom_num_samples, nystrom_sigma, nystrom_num_clusters);

    disp('saving spectral clustering results using Nystrom methods ...');
    if exist(resultant_clustering_file_path, 'file')
        save(resultant_clustering_file_path, 'nystrom_cluster_labels', 'nystrom_no_orth_cluster_labels', '-append');
    else
        save(resultant_clustering_file_path, 'nystrom_cluster_labels', 'nystrom_no_orth_cluster_labels');
    end;
end;

if does_run_kmeans_algorithm
    disp('running k-means clustering ...');
    kmeans_cluster_labels = k_means(HoG_sequences2', kmeans_centers, kmeans_num_clusters);

    disp('saving clustering results of k-means algorithm ...');
    if exist(resultant_clustering_file_path, 'file')
        save(resultant_clustering_file_path, 'kmeans_cluster_labels', '-append');
    else
        save(resultant_clustering_file_path, 'kmeans_cluster_labels');
    end;
end;
