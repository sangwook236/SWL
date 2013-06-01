kinect_ir_file_names = {
	'kinect_ir_histeq_01.png',
	'kinect_ir_histeq_02.png',
	'kinect_ir_histeq_03.png',
	'kinect_ir_histeq_04.png',
	'kinect_ir_histeq_05.png',
	'kinect_ir_histeq_06.png',
	'kinect_ir_histeq_07.png',
	'kinect_ir_histeq_08.png',
	'kinect_ir_histeq_09.png',
	'kinect_ir_histeq_10.png',
	'kinect_ir_histeq_11.png',
	'kinect_ir_histeq_12.png',
	'kinect_ir_histeq_13.png',
	'kinect_ir_histeq_14.png',
	'kinect_ir_histeq_15.png',
	'kinect_ir_histeq_16.png'
};

num_files = length(kinect_ir_file_names);

bin_threshold = 0.65 * ones(1, num_files);
bin_threshold(5) = 0.64;
bin_threshold(6) = 0.63;
bin_threshold(7) = 0.56;
bin_threshold(11) = 0.64;
bin_threshold(12) = 0.6;
bin_threshold(15) = 0.67;

%figure;
for ii = 1:num_files
	img = imread(kinect_ir_file_names{ii});

	%img = rgb2gray(img);
	%img = histeq(img);

	img = im2bw(img, bin_threshold(ii));
	img = medfilt2(img);

	imwrite(img, kinect_ir_file_names{ii});
	%imshow(img)
end;
