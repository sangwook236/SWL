% Intrinsic and Extrinsic Camera Parameters
%
% This script file can be directly excecuted under Matlab to recover the camera intrinsic and extrinsic parameters.
% IMPORTANT: This file contains neither the structure of the calibration objects nor the image coordinates of the calibration points.
%            All those complementary variables are saved in the complete matlab data file Calib_Results.mat.
% For more information regarding the calibration model visit http://www.vision.caltech.edu/bouguetj/calib_doc/


%-- Focal length:
fc = [ 585.725110330112440 ; 586.150984962782330 ];

%-- Principal point:
cc = [ 336.039644006935020 ; 246.843007895227660 ];

%-- Skew coefficient:
alpha_c = 0.000000000000000;

%-- Distortion coefficients:
kc = [ -0.111314439869815 ; 0.390204235494320 ; -0.002473313414950 ; 0.006053929513996 ; -0.234253519748674 ];

%-- Focal length uncertainty:
fc_error = [ 11.106319209078606 ; 10.513197929644504 ];

%-- Principal point uncertainty:
cc_error = [ 17.952266176913049 ; 15.022955452699867 ];

%-- Skew coefficient uncertainty:
alpha_c_error = 0.000000000000000;

%-- Distortion coefficients uncertainty:
kc_error = [ 0.160659261751773 ; 1.739318243077645 ; 0.008242993018329 ; 0.010116600472644 ; 5.569096099753273 ];

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
est_dist = [ 1 ; 1 ; 1 ; 1 ; 1 ];	% Estimation indicator of the distortion coefficients


%-- Extrinsic parameters:
%-- The rotation (omc_kk) and the translation (Tc_kk) vectors for every calibration image and their uncertainties

%-- Image #1:
omc_1 = [ -1.978396e+00 ; -1.995922e+00 ; 1.287272e-01 ];
Tc_1  = [ -2.358584e+02 ; -2.251722e+02 ; 1.486343e+03 ];
omc_error_1 = [ 2.507712e-02 ; 2.680028e-02 ; 5.194149e-02 ];
Tc_error_1  = [ 4.591765e+01 ; 3.830169e+01 ; 2.793095e+01 ];

%-- Image #2:
omc_2 = [ 2.107317e+00 ; 1.805089e+00 ; 7.407343e-01 ];
Tc_2  = [ -5.099373e+02 ; -2.356426e+02 ; 1.529295e+03 ];
omc_error_2 = [ 2.715403e-02 ; 2.172975e-02 ; 4.673921e-02 ];
Tc_error_2  = [ 4.932024e+01 ; 4.081310e+01 ; 3.617183e+01 ];

%-- Image #3:
omc_3 = [ -1.993238e+00 ; -1.904185e+00 ; -3.646414e-01 ];
Tc_3  = [ -1.968959e+02 ; -5.260426e+02 ; 1.822379e+03 ];
omc_error_3 = [ 3.038922e-02 ; 3.213809e-02 ; 5.876109e-02 ];
Tc_error_3  = [ 5.751831e+01 ; 4.871479e+01 ; 3.982116e+01 ];

%-- Image #4:
omc_4 = [ 2.114677e+00 ; 1.521890e+00 ; 1.132271e+00 ];
Tc_4  = [ -5.499952e+02 ; -1.735446e+02 ; 1.264794e+03 ];
omc_error_4 = [ 2.905897e-02 ; 1.717036e-02 ; 3.963955e-02 ];
Tc_error_4  = [ 4.178313e+01 ; 3.453412e+01 ; 3.233525e+01 ];

%-- Image #5:
omc_5 = [ -1.570334e+00 ; -1.804537e+00 ; 2.429175e-01 ];
Tc_5  = [ -4.329179e+02 ; -2.581642e+02 ; 1.428421e+03 ];
omc_error_5 = [ 2.432593e-02 ; 2.582349e-02 ; 3.709143e-02 ];
Tc_error_5  = [ 4.470918e+01 ; 3.752475e+01 ; 2.804698e+01 ];

%-- Image #6:
omc_6 = [ -2.012004e+00 ; -1.856279e+00 ; -4.365399e-01 ];
Tc_6  = [ -2.613212e+02 ; -1.371535e+02 ; 9.949908e+02 ];
omc_error_6 = [ 1.838699e-02 ; 2.517943e-02 ; 4.315605e-02 ];
Tc_error_6  = [ 3.083833e+01 ; 2.597542e+01 ; 2.154889e+01 ];

%-- Image #7:
omc_7 = [ -1.545670e+00 ; -1.790615e+00 ; -2.621578e-01 ];
Tc_7  = [ -5.282773e+02 ; -2.271719e+02 ; 1.274534e+03 ];
omc_error_7 = [ 2.217794e-02 ; 2.629704e-02 ; 3.724449e-02 ];
Tc_error_7  = [ 3.973104e+01 ; 3.419554e+01 ; 3.082101e+01 ];

%-- Image #8:
omc_8 = [ -2.089323e+00 ; -1.694629e+00 ; -1.027990e+00 ];
Tc_8  = [ -4.890880e+02 ; -7.833596e+01 ; 1.106274e+03 ];
omc_error_8 = [ 2.157586e-02 ; 2.403459e-02 ; 4.263602e-02 ];
Tc_error_8  = [ 3.454775e+01 ; 3.005261e+01 ; 2.934096e+01 ];

%-- Image #9:
omc_9 = [ -2.076679e+00 ; -1.650107e+00 ; -1.110755e+00 ];
Tc_9  = [ -1.614689e+02 ; -1.194072e+02 ; 1.087795e+03 ];
omc_error_9 = [ 1.723184e-02 ; 2.719209e-02 ; 4.324799e-02 ];
Tc_error_9  = [ 3.368568e+01 ; 2.813645e+01 ; 2.416211e+01 ];

%-- Image #10:
omc_10 = [ 2.070877e+00 ; 2.010059e+00 ; -4.082883e-01 ];
Tc_10  = [ -3.522632e+02 ; -2.833971e+02 ; 1.335251e+03 ];
omc_error_10 = [ 1.969998e-02 ; 2.833636e-02 ; 4.849945e-02 ];
Tc_error_10  = [ 4.157436e+01 ; 3.489712e+01 ; 2.655708e+01 ];

%-- Image #11:
omc_11 = [ -1.862676e+00 ; -1.699953e+00 ; 1.161981e+00 ];
Tc_11  = [ -2.726805e+01 ; -2.274063e+02 ; 1.597582e+03 ];
omc_error_11 = [ 3.056382e-02 ; 1.656090e-02 ; 3.857649e-02 ];
Tc_error_11  = [ 4.935341e+01 ; 4.080461e+01 ; 2.324036e+01 ];

%-- Image #12:
omc_12 = [ 1.745218e+00 ; 1.789894e+00 ; 2.677793e-01 ];
Tc_12  = [ -1.104414e+02 ; -4.026679e+02 ; 1.102691e+03 ];
omc_error_12 = [ 2.229768e-02 ; 2.609048e-02 ; 3.781103e-02 ];
Tc_error_12  = [ 3.492944e+01 ; 2.838433e+01 ; 2.358060e+01 ];

%-- Image #13:
omc_13 = [ 1.762934e+00 ; 1.795620e+00 ; 2.388446e-01 ];
Tc_13  = [ -6.178811e+02 ; -1.806744e+02 ; 2.148789e+03 ];
omc_error_13 = [ 2.301728e-02 ; 2.894496e-02 ; 4.766891e-02 ];
Tc_error_13  = [ 6.785747e+01 ; 5.657148e+01 ; 4.883959e+01 ];

%-- Image #14:
omc_14 = [ -2.118364e+00 ; -2.008795e+00 ; 8.586863e-01 ];
Tc_14  = [ -9.328657e+01 ; -6.228447e+01 ; 2.427212e+03 ];
omc_error_14 = [ 3.356109e-02 ; 2.491778e-02 ; 5.399498e-02 ];
Tc_error_14  = [ 7.438423e+01 ; 6.220206e+01 ; 4.135987e+01 ];

%-- Image #15:
omc_15 = [ -1.940003e+00 ; -2.066511e+00 ; -1.490860e-01 ];
Tc_15  = [ 1.276696e+02 ; -2.972592e+02 ; 2.055387e+03 ];
omc_error_15 = [ 3.049949e-02 ; 4.256796e-02 ; 6.994192e-02 ];
Tc_error_15  = [ 6.340481e+01 ; 5.313095e+01 ; 4.548114e+01 ];

%-- Image #16:
omc_16 = [ 2.190942e+00 ; 2.109656e+00 ; 6.408744e-01 ];
Tc_16  = [ -1.774176e+02 ; -3.305884e+02 ; 2.108885e+03 ];
omc_error_16 = [ 3.442675e-02 ; 2.871851e-02 ; 6.059705e-02 ];
Tc_error_16  = [ 6.533208e+01 ; 5.445720e+01 ; 4.813240e+01 ];

