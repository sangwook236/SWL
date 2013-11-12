addpath('E:\work_center\sw_dev\matlab\rnd\src\prtools\prtools_ac\prtools');
addpath('E:\work_center\sw_dev\matlab\rnd\src\dimensionality_reduction\Matlab_Toolbox_for_Dimensionality_Reduction\drtoolbox');

img_row = 640;
img_col = 486;

% process raw images
raw_img_parent_dir_pathname = '..\data\cmu_mobo\moboJpg\04077\fastWalk';
bg_img_parent_dir_pathname = '..\data\cmu_mobo\moboJpg\04077\bgImage';
raw_img_dir_name = '\vr05_7';
bg_img_file_name = 'backgroundImage22_00481523.jpg';

raw_img_dir_pathname = strcat(strcat(raw_img_parent_dir_pathname, raw_img_dir_name));
bg_img_dir_pathname = strcat(strcat(bg_img_parent_dir_pathname, raw_img_dir_name));
bg_img = imread(strcat(strcat(bg_img_dir_pathname, '\'), bg_img_file_name));

color_img = imread(strcat(strcat(raw_img_dir_pathname, '\'), 'im22_00450200.jpg'));

figure; imshow(imabsdiff(color_img, bg_img));
figure; imshow(imsubtract(color_img, bg_img));
