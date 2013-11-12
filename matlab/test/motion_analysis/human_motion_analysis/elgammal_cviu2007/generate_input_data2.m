function [ X ] = generate_input_data2(raw_img_dir_pathname, img_row, img_col)

img_len = img_row * img_col;

raw_img_files = dir(strcat(raw_img_dir_pathname, '\*.pbm'));
img_num = length(raw_img_files);
X = zeros(img_num, img_len);

for kk = 1:img_num
    img = imread(strcat(strcat(raw_img_dir_pathname, '\'), raw_img_files(kk).name));
    if img_row ~= size(img, 1) || img_col ~= size(img, 2)
        img = imresize(img, [ img_row, img_col ]);
    end;

    X(kk,:) = reshape(img, 1, img_len);
end;

save(strcat(raw_img_dir_pathname, '\imgX2.mat'), 'X');
