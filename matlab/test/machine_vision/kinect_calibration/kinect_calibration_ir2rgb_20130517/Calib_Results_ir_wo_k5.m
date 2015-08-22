% Intrinsic and Extrinsic Camera Parameters
%
% This script file can be directly excecuted under Matlab to recover the camera intrinsic and extrinsic parameters.
% IMPORTANT: This file contains neither the structure of the calibration objects nor the image coordinates of the calibration points.
%            All those complementary variables are saved in the complete matlab data file Calib_Results.mat.
% For more information regarding the calibration model visit http://www.vision.caltech.edu/bouguetj/calib_doc/


%-- Focal length:
fc = [ 585.753592247520710 ; 586.570803070341190 ];

%-- Principal point:
cc = [ 335.193217452468450 ; 246.416568443205930 ];

%-- Skew coefficient:
alpha_c = 0.000000000000000;

%-- Distortion coefficients:
kc = [ -0.106390158049948 ; 0.339519288181204 ; -0.002211031053332 ; 0.005882227715342 ; -0.000000000000000 ];

%-- Focal length uncertainty:
fc_error = [ 10.985262135526416 ; 10.443468111107361 ];

%-- Principal point uncertainty:
cc_error = [ 17.343188791056583 ; 14.976775159856587 ];

%-- Skew coefficient uncertainty:
alpha_c_error = 0.000000000000000;

%-- Distortion coefficients uncertainty:
kc_error = [ 0.070161945234210 ; 0.350774923994858 ; 0.008274497120155 ; 0.009962744633119 ; 0.000000000000000 ];

%-- Image size:
nx = 640;
ny = 480;


%-- Various other variables (may be ignored if you do not use the Matlab Calibration Toolbox):
%-- Those variables are used to control which intrinsic parameters should be optimized

n_ima = 16;						% Number of calibration images
est_fc = [ 1 ; 1 ];					% Estimation indicator of the two focal variables
est_aspect_ratio = 1;				% Estimation indicator of the aspect ratio fc(2)/fc(1)
center_optim = 1;					% Estimation indicator of the principal point
est_alpha = 0;						% Estimation indicator of the skew coefficient
est_dist = [ 1 ; 1 ; 1 ; 1 ; 0 ];	% Estimation indicator of the distortion coefficients


%-- Extrinsic parameters:
%-- The rotation (omc_kk) and the translation (Tc_kk) vectors for every calibration image and their uncertainties

%-- Image #1:
omc_1 = [ -1.978539e+00 ; -1.995588e+00 ; 1.285659e-01 ];
Tc_1  = [ -2.337915e+02 ; -2.240743e+02 ; 1.487181e+03 ];
omc_error_1 = [ 2.475492e-02 ; 2.643280e-02 ; 5.159958e-02 ];
Tc_error_1  = [ 4.440312e+01 ; 3.817200e+01 ; 2.773956e+01 ];

%-- Image #2:
omc_2 = [ 2.109106e+00 ; 1.806823e+00 ; 7.376511e-01 ];
Tc_2  = [ -5.081865e+02 ; -2.345213e+02 ; 1.531914e+03 ];
omc_error_2 = [ 2.663776e-02 ; 2.139353e-02 ; 4.631557e-02 ];
Tc_error_2  = [ 4.779584e+01 ; 4.070915e+01 ; 3.585926e+01 ];

%-- Image #3:
omc_3 = [ -1.992673e+00 ; -1.903293e+00 ; -3.637769e-01 ];
Tc_3  = [ -1.942153e+02 ; -5.244907e+02 ; 1.823139e+03 ];
omc_error_3 = [ 3.006382e-02 ; 3.162081e-02 ; 5.831938e-02 ];
Tc_error_3  = [ 5.560277e+01 ; 4.852352e+01 ; 3.970741e+01 ];

%-- Image #4:
omc_4 = [ 2.115894e+00 ; 1.523005e+00 ; 1.129588e+00 ];
Tc_4  = [ -5.484963e+02 ; -1.726940e+02 ; 1.266768e+03 ];
omc_error_4 = [ 2.834131e-02 ; 1.672609e-02 ; 3.921244e-02 ];
Tc_error_4  = [ 4.050048e+01 ; 3.443266e+01 ; 3.194377e+01 ];

%-- Image #5:
omc_5 = [ -1.570396e+00 ; -1.803511e+00 ; 2.438774e-01 ];
Tc_5  = [ -4.310271e+02 ; -2.571131e+02 ; 1.429734e+03 ];
omc_error_5 = [ 2.397401e-02 ; 2.496700e-02 ; 3.658954e-02 ];
Tc_error_5  = [ 4.324598e+01 ; 3.740462e+01 ; 2.768840e+01 ];

%-- Image #6:
omc_6 = [ -2.011729e+00 ; -1.855375e+00 ; -4.357504e-01 ];
Tc_6  = [ -2.598752e+02 ; -1.363416e+02 ; 9.956047e+02 ];
omc_error_6 = [ 1.823043e-02 ; 2.456375e-02 ; 4.239603e-02 ];
Tc_error_6  = [ 2.981796e+01 ; 2.588085e+01 ; 2.132969e+01 ];

%-- Image #7:
omc_7 = [ -1.545856e+00 ; -1.789554e+00 ; -2.610196e-01 ];
Tc_7  = [ -5.265114e+02 ; -2.261897e+02 ; 1.275729e+03 ];
omc_error_7 = [ 2.204008e-02 ; 2.555356e-02 ; 3.656555e-02 ];
Tc_error_7  = [ 3.842465e+01 ; 3.407947e+01 ; 3.033147e+01 ];

%-- Image #8:
omc_8 = [ -2.089491e+00 ; -1.694253e+00 ; -1.025396e+00 ];
Tc_8  = [ -4.876002e+02 ; -7.754845e+01 ; 1.107584e+03 ];
omc_error_8 = [ 2.149439e-02 ; 2.373580e-02 ; 4.154383e-02 ];
Tc_error_8  = [ 3.342972e+01 ; 2.994995e+01 ; 2.889835e+01 ];

%-- Image #9:
omc_9 = [ -2.077224e+00 ; -1.650099e+00 ; -1.108259e+00 ];
Tc_9  = [ -1.599022e+02 ; -1.186162e+02 ; 1.088774e+03 ];
omc_error_9 = [ 1.715993e-02 ; 2.687870e-02 ; 4.222680e-02 ];
Tc_error_9  = [ 3.257150e+01 ; 2.804366e+01 ; 2.405082e+01 ];

%-- Image #10:
omc_10 = [ 2.070122e+00 ; 2.009917e+00 ; -4.108069e-01 ];
Tc_10  = [ -3.503340e+02 ; -2.823897e+02 ; 1.336285e+03 ];
omc_error_10 = [ 1.962035e-02 ; 2.791690e-02 ; 4.770904e-02 ];
Tc_error_10  = [ 4.020786e+01 ; 3.477708e+01 ; 2.622551e+01 ];

%-- Image #11:
omc_11 = [ -1.862143e+00 ; -1.699423e+00 ; 1.163828e+00 ];
Tc_11  = [ -2.494752e+01 ; -2.262406e+02 ; 1.598766e+03 ];
omc_error_11 = [ 2.971766e-02 ; 1.621548e-02 ; 3.804196e-02 ];
Tc_error_11  = [ 4.771388e+01 ; 4.067657e+01 ; 2.316589e+01 ];

%-- Image #12:
omc_12 = [ 1.745691e+00 ; 1.790436e+00 ; 2.665862e-01 ];
Tc_12  = [ -1.089115e+02 ; -4.018088e+02 ; 1.103278e+03 ];
omc_error_12 = [ 2.204257e-02 ; 2.530464e-02 ; 3.728768e-02 ];
Tc_error_12  = [ 3.376778e+01 ; 2.828826e+01 ; 2.343829e+01 ];

%-- Image #13:
omc_13 = [ 1.763147e+00 ; 1.796421e+00 ; 2.352315e-01 ];
Tc_13  = [ -6.150765e+02 ; -1.791035e+02 ; 2.151229e+03 ];
omc_error_13 = [ 2.281983e-02 ; 2.855475e-02 ; 4.737451e-02 ];
Tc_error_13  = [ 6.563497e+01 ; 5.640325e+01 ; 4.845371e+01 ];

%-- Image #14:
omc_14 = [ -2.117461e+00 ; -2.008505e+00 ; 8.630550e-01 ];
Tc_14  = [ -8.966186e+01 ; -6.058942e+01 ; 2.427550e+03 ];
omc_error_14 = [ 3.293580e-02 ; 2.460897e-02 ; 5.337984e-02 ];
Tc_error_14  = [ 7.186627e+01 ; 6.196652e+01 ; 4.119913e+01 ];

%-- Image #15:
omc_15 = [ -1.946573e+00 ; -2.068274e+00 ; -1.861134e-01 ];
Tc_15  = [ 1.301596e+02 ; -2.944908e+02 ; 2.047790e+03 ];
omc_error_15 = [ 2.917114e-02 ; 4.079505e-02 ; 6.750512e-02 ];
Tc_error_15  = [ 6.105431e+01 ; 5.270108e+01 ; 4.530465e+01 ];

%-- Image #16:
omc_16 = [ 2.192594e+00 ; 2.111835e+00 ; 6.429009e-01 ];
Tc_16  = [ -1.741811e+02 ; -3.285601e+02 ; 2.107792e+03 ];
omc_error_16 = [ 3.383666e-02 ; 2.822063e-02 ; 5.974166e-02 ];
Tc_error_16  = [ 6.311079e+01 ; 5.421687e+01 ; 4.788517e+01 ];

