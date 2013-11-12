addpath('E:\work_center\sw_dev\matlab\rnd\src\prtools\prtools_ac\prtools');
addpath('E:\work_center\sw_dev\matlab\rnd\src\dimensionality_reduction\Matlab_Toolbox_for_Dimensionality_Reduction\drtoolbox');
addpath('E:\work_center\sw_dev\matlab\rnd\src\dimensionality_reduction\Matlab_Toolbox_for_Dimensionality_Reduction\drtoolbox\techniques');

img_row = 640;
img_col = 486;

% process raw images
raw_img_parent_dir_pathname = '..\..\data\cmu_mobo\moboBgSub\04077\fastWalk';
raw_img_dir_name = '\vr05_7';

%se_shape = 'square';
%se_shape = 'diamond';
se_shape = 'disk';
se_radius = 7;

%processed_raw_img_dir_pathname = process_raw_images(raw_img_parent_dir_pathname, raw_img_dir_name, se_shape, se_radius);
processed_raw_img_dir_pathname = '..\..\data\cmu_mobo\moboBgSub\04077\fastWalk\vr05_7_disk07';

% generate input images
img_scale = 0.5;
img_representation_method = 1;
load_mat_data = 1;
switch img_representation_method
    case 1
        % vectorize each raw image with distance to boundary
        if 1 == load_mat_data
            load(strcat(processed_raw_img_dir_pathname, '\imgX1.mat'), 'X');
        else
            [ X ] = generate_input_data1(processed_raw_img_dir_pathname, img_row * img_scale, img_col * img_scale);
        end;
    case 2
        % vectorize each raw image with intensity 0/1
        if 1 == load_mat_data
            load(strcat(processed_raw_img_dir_pathname, '\imgX2.mat'), 'X');
        else
            [ X ] = generate_input_data2(processed_raw_img_dir_pathname, img_row * img_scale, img_col * img_scale);
        end;
    case 3
        % generate mhi & vectorize each raw image
        if 1 == load_mat_data
            load(strcat(processed_raw_img_dir_pathname, '\imgX_mhi.mat'), 'X');
        else
            stack_img_num = 20;
            [ X ] = generate_mhi(processed_raw_img_dir_pathname, img_row * img_scale, img_col * img_scale, stack_img_num);
        end;
end

%
img_num = size(X, 1);
img_size = size(X, 2);

use_centered_data = 1;
use_moment_feature = 0;
max_moment_order = 10;
if 1 == use_moment_feature
    mom = zeros(img_num, max_moment_order);
    for kk = 1:max_moment_order
        mom(:,kk) = moment(X, kk, 2);
    end
    X = mom;
elseif 1 == use_centered_data
    X_ave = mean(X, 1);
    for kk = 1:img_num
        X(kk,:) = X(kk,:) - X_ave;
    end;
end;

% compute embedding
data_intrinsic_dim = round(intrinsic_dim(X, 'MLE'));
[Ye, Me] = compute_mapping(X, 'LLE', data_intrinsic_dim);
% [Ye, Me] = compute_mapping(X, 'LLE', 3);
% [Ye, Me] = compute_mapping(X, 'LTSA', data_intrinsic_dim, 12, 'JDQR');
% figure;
% plot3(Ye(:,1), Ye(:,2), Ye(:,3), '.-');

if size(X,1) ~= size(Ye,1)
    X = X(Me.conn_comp,:);
end;

% RBF interpolant
N = size(Ye, 1);
d = size(X, 2);
e = data_intrinsic_dim;
%Nt = N;
Nt = 24;

phi = inline('x^2 * log(x)');  % basis function

if Nt == N
    P = ones(N, 1 + e);
    P(:,2:end) = Ye;
    C = zeros(N + 1 + e, d);
    C(1:N,:) = X;

    A = zeros(N, N);
    for ii = 1:N
        for jj = 1:N
            % FIXME [check] >> 2-norm ???
            nn = norm(Ye(jj,:)-Ye(ii,:), 2);
            if nn < 1.0e-20
                % FIXME [check] >>
                A(ii,jj) = 0;
            else
                A(ii,jj) = phi(nn);
            end;
        end;
    end;

    B = ([ A P ; P' zeros(e+1, e+1) ] \ C)';
else
    rmpath('E:\work_center\sw_dev\matlab\rnd\src\prtools\prtools_ac\prtools');
    [IDX, Te] = kmeans(Ye, Nt);  % when using matlab function
    addpath('E:\work_center\sw_dev\matlab\rnd\src\prtools\prtools_ac\prtools');

    Px = ones(N, 1 + e);
    Px(:,2:end) = Ye;
    Pt = ones(Nt, 1 + e);
    Pt(:,2:end) = Te;
    C = zeros(N + 1 + e, d);
    C(1:N,:) = X;

    A = zeros(N, Nt);
    for ii = 1:N
        for jj = 1:Nt
            % FIXME [check] >> 2-norm ???
            nn = norm(Te(jj,:)-Ye(ii,:), 2);
            if nn < 1.0e-20
                % FIXME [check] >>
                A(ii,jj) = 0;
            else
                A(ii,jj) = phi(nn);
            end;
        end;
    end;

    B = ([ A Px ; Pt' zeros(e+1, e+1) ] \ C)';
end;

[U, S, V] = svd(B, 'econ');
invB = V * pinv(S) * U';

% solving for the embedding coordinates
test_case_index = 2;
if 1 == test_case_index
    x_new = X(1,:)';
elseif 2 == test_case_index
    test_img_parent_dir_pathname = raw_img_parent_dir_pathname;
    test_img_dir_name = raw_img_dir_name;
    test_img_file_name = '\sil10001.pbm';
    x_new_img = imread(strcat(strcat(test_img_parent_dir_pathname, test_img_dir_name), test_img_file_name));
    if img_row*img_scale ~= size(x_new_img, 1) || img_col*img_scale ~= size(x_new_img, 2)
        x_new_img = imresize(x_new_img, [ img_row*img_scale, img_col*img_scale ]);
    end;
    x_new = x_new_img(:);
    if 1 == use_centered_data
        x_new = x_new - X_ave';
    end;
elseif 3 == test_case_index
    test_img_parent_dir_pathname = '..\..\data\cmu_mobo\moboBgSub\04072\fastWalk';
    test_img_dir_name = '\vr05_7';
    test_img_file_name = '\sil10001.pbm';
    x_new_img = imread(strcat(strcat(test_img_parent_dir_pathname, test_img_dir_name), test_img_file_name));
    if img_row*img_scale ~= size(x_new_img, 1) || img_col*img_scale ~= size(x_new_img, 2)
        x_new_img = imresize(x_new_img, [ img_row*img_scale, img_col*img_scale ]);
    end;
    x_new = x_new_img(:);
    if 1 == use_centered_data
        x_new = x_new - X_ave';
    end;
end;

psi = invB * x_new;
if Nt == N
    %y_new_hat = psi(N+2:end,:);
    %x_new_hat = B * [zeros(N,1) ; 1 ; y_new_hat];
    x_new_hat = B * psi;
else
    y_new_hat = psi(Nt+2:end,:);
    x_new_hat = B * [zeros(Nt,1) ; 1 ; y_new_hat];
    %x_new_hat = B * psi;
end;

if 1 ~= use_moment_feature
    if 1 == use_centered_data
        imshow(reshape(x_new_hat + X_ave' >= 0, [img_row*img_scale img_col*img_scale]));
    else
        imshow(reshape(x_new_hat >= 0, [img_row*img_scale img_col*img_scale]));
    end;
end;
