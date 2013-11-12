function [ processed_raw_img_dir_pathname ] = process_raw_images(raw_img_parent_dir_pathname, raw_img_dir_name, se_shape, se_radius)

raw_img_dir_pathname = strcat(raw_img_parent_dir_pathname, raw_img_dir_name);
raw_img_files = dir(strcat(raw_img_dir_pathname, '\*.pbm'));
img_num = length(raw_img_files);

dir_tag = sprintf('_%s%02d', se_shape, se_radius);
processed_raw_img_dir_name = strcat(raw_img_dir_name, dir_tag);
processed_raw_img_dir_pathname = strcat(raw_img_parent_dir_pathname, processed_raw_img_dir_name);
if 7 ~= exist(processed_raw_img_dir_pathname, 'dir')
    mkdir(raw_img_parent_dir_pathname, processed_raw_img_dir_name);
end;

imgs = cell(img_num, 1);
for ii = 1:img_num
    imgs{ii} = imread(strcat(strcat(raw_img_dir_pathname, '\'), raw_img_files(ii).name));

    se = strel(se_shape, se_radius);

    imgs{ii} = imdilate(imgs{ii}, se);
    imgs{ii} = imerode(imgs{ii}, se);
    %imopen(img, se);
    %imclose(img, se);
    %figure; imshow(imgs(ii));
    
    imwrite(imgs{ii}, strcat(strcat(processed_raw_img_dir_pathname, '\'), raw_img_files(ii).name));
end;
