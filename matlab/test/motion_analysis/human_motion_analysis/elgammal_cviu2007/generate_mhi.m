function [ X ] = generate_mhi(raw_img_dir_pathname, img_row, img_col, stack_img_num)

img_len = img_row * img_col;

raw_img_files = dir(strcat(raw_img_dir_pathname, '\*.pbm'));
img_num = length(raw_img_files);
X = zeros(img_num - stack_img_num + 1, img_len);

mhi_coeffs = linspace(1, 0, stack_img_num + 1);
img_stack_idx = 1;
img_stack = cell(1, stack_img_num);
mhi = zeros(img_row, img_col);

ll = 1;
for kk = 1:img_num
    img = imread(strcat(strcat(raw_img_dir_pathname, '\'), raw_img_files(kk).name));
    if img_row ~= size(img, 1) || img_col ~= size(img, 2)
        img = imresize(img, [ img_row, img_col ]);
    end;
    
    img_stack{img_stack_idx} = img;
    img_stack_idx = rem(img_stack_idx, stack_img_num) + 1;

    if kk < stack_img_num
        continue;
    end;
    
    jj = stack_img_num;
    mhi = img_stack{jj};
    for ii = stack_img_num-1:-1:1
        jj = jj - 1;
        if 0 == jj
            jj = stack_img_num;
        end;
        mhi = mhi + mhi_coeffs(ii+1) * img_stack{jj};
    end;

    X(ll,:) = reshape(mhi, 1, img_len);
    ll = ll + 1;
end;

save(strcat(raw_img_dir_pathname, '\imgX_mhi.mat'), 'X');
