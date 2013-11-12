function [ X ] = generate_input_data1(raw_img_dir_pathname, img_row, img_col)

% [ref]
% "Nonlinear manifold learning for dynamic shape and dynamic appearance",
% Ahmed Elgammal & Chan-Su Lee, CVIU 2007, pp. 33~34

img_len = img_row * img_col;

raw_img_files = dir(strcat(raw_img_dir_pathname, '\*.pbm'));
img_num = length(raw_img_files);
X = zeros(img_num, img_len);

% 'chessboard', 'cityblock', 'euclidean', 'quasi-euclidean'
dist_measure = 'euclidean';

for kk = 1:img_num
    img = imread(strcat(strcat(raw_img_dir_pathname, '\'), raw_img_files(kk).name));
    if img_row ~= size(img, 1) || img_col ~= size(img, 2)
        img = imresize(img, [ img_row, img_col ]);
    end;

	% compute the distance transform of a binary image
    D1 = bwdist(img, dist_measure);

    img2 = img;
%     for ii = 1:img_row
%         for jj = 1:img_col
%             pixel = img(ii,jj);
%             if 1 == pixel
%                 if (img(max(ii-1,1),jj) ~= pixel) || (img(min(ii+1,img_row),jj) ~= pixel) || ...
%                     (img(ii,max(jj-1,1)) ~= pixel) || (img(ii,min(jj+1,img_col)) ~= pixel)
%                     img2(ii,jj) = 0;
%                 end;
%             end;
%         end;
%     end;
    boundaries = bwboundaries(img2);
    for mm = 1:size(boundaries,1)
        b = boundaries{mm};
        for ii = 1:size(b,1)
            img2(b(ii,1),b(ii,2)) = 0;
        end;
    end;

	% compute the distance transform of a binary image
    D2 = bwdist(1 - img2, dist_measure);

    X(kk,:) = reshape(D2 - D1, 1, img_len);
end;

save(strcat(raw_img_dir_pathname, '\imgX1.mat'), 'X');
