% Intrinsic and Extrinsic Camera Parameters
%
% This script file can be directly excecuted under Matlab to recover the camera intrinsic and extrinsic parameters.
% IMPORTANT: This file contains neither the structure of the calibration objects nor the image coordinates of the calibration points.
%            All those complementary variables are saved in the complete matlab data file Calib_Results.mat.
% For more information regarding the calibration model visit http://www.vision.caltech.edu/bouguetj/calib_doc/


%-- Focal length:
fc = [ 526.176912808111750 ; 528.069366896795260 ];

%-- Principal point:
cc = [ 329.021564996589230 ; 265.146285733477040 ];

%-- Skew coefficient:
alpha_c = 0.000000000000000;

%-- Distortion coefficients:
kc = [ 0.263971723688510 ; -0.902637692213340 ; 0.002569103898876 ; 0.004773654687023 ; 1.074728662132601 ];

%-- Focal length uncertainty:
fc_error = [ 2.965813151895432 ; 2.835063888701568 ];

%-- Principal point uncertainty:
cc_error = [ 3.869998610642791 ; 3.865818331603654 ];

%-- Skew coefficient uncertainty:
alpha_c_error = 0.000000000000000;

%-- Distortion coefficients uncertainty:
kc_error = [ 0.049245393787069 ; 0.536868695369863 ; 0.003695446324407 ; 0.003402588628095 ; 1.746226750789846 ];

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
omc_1 = [ -1.973149e+00 ; -1.995550e+00 ; 1.190153e-01 ];
Tc_1  = [ -2.001072e+02 ; -2.320826e+02 ; 1.483275e+03 ];
omc_error_1 = [ 6.727607e-03 ; 7.148781e-03 ; 1.402693e-02 ];
Tc_error_1  = [ 1.088831e+01 ; 1.085908e+01 ; 8.294830e+00 ];

%-- Image #2:
omc_2 = [ 2.116865e+00 ; 1.814745e+00 ; 7.228640e-01 ];
Tc_2  = [ -4.757592e+02 ; -2.454450e+02 ; 1.537321e+03 ];
omc_error_2 = [ 7.942521e-03 ; 6.392747e-03 ; 1.340617e-02 ];
Tc_error_2  = [ 1.163780e+01 ; 1.157484e+01 ; 1.064338e+01 ];

%-- Image #3:
omc_3 = [ -1.990158e+00 ; -1.914942e+00 ; -3.293799e-01 ];
Tc_3  = [ -1.562888e+02 ; -5.358541e+02 ; 1.824568e+03 ];
omc_error_3 = [ 9.346798e-03 ; 9.763319e-03 ; 1.755003e-02 ];
Tc_error_3  = [ 1.370555e+01 ; 1.381536e+01 ; 1.183250e+01 ];

%-- Image #4:
omc_4 = [ 2.123712e+00 ; 1.527656e+00 ; 1.129060e+00 ];
Tc_4  = [ -5.172647e+02 ; -1.819672e+02 ; 1.271052e+03 ];
omc_error_4 = [ 7.960948e-03 ; 4.745166e-03 ; 1.038817e-02 ];
Tc_error_4  = [ 9.878776e+00 ; 9.805986e+00 ; 9.473212e+00 ];

%-- Image #5:
omc_5 = [ -1.564152e+00 ; -1.802766e+00 ; 2.411734e-01 ];
Tc_5  = [ -3.978900e+02 ; -2.660585e+02 ; 1.430584e+03 ];
omc_error_5 = [ 6.506691e-03 ; 6.054236e-03 ; 9.294129e-03 ];
Tc_error_5  = [ 1.055404e+01 ; 1.062721e+01 ; 7.984580e+00 ];

%-- Image #6:
omc_6 = [ -2.007106e+00 ; -1.852618e+00 ; -4.317933e-01 ];
Tc_6  = [ -2.284603e+02 ; -1.368080e+02 ; 9.955673e+02 ];
omc_error_6 = [ 4.945328e-03 ; 6.708828e-03 ; 1.058093e-02 ];
Tc_error_6  = [ 7.340223e+00 ; 7.338047e+00 ; 6.125571e+00 ];

%-- Image #7:
omc_7 = [ -1.539083e+00 ; -1.785450e+00 ; -2.625113e-01 ];
Tc_7  = [ -4.936544e+02 ; -2.338038e+02 ; 1.275409e+03 ];
omc_error_7 = [ 6.125287e-03 ; 6.459909e-03 ; 9.189867e-03 ];
Tc_error_7  = [ 9.417838e+00 ; 9.688744e+00 ; 8.649232e+00 ];

%-- Image #8:
omc_8 = [ -2.083033e+00 ; -1.689186e+00 ; -1.021452e+00 ];
Tc_8  = [ -4.561965e+02 ; -8.493059e+01 ; 1.108748e+03 ];
omc_error_8 = [ 5.871960e-03 ; 6.877045e-03 ; 1.045900e-02 ];
Tc_error_8  = [ 8.245889e+00 ; 8.501772e+00 ; 8.260782e+00 ];

%-- Image #9:
omc_9 = [ -2.073109e+00 ; -1.650260e+00 ; -1.100148e+00 ];
Tc_9  = [ -1.287493e+02 ; -1.264074e+02 ; 1.092807e+03 ];
omc_error_9 = [ 4.748320e-03 ; 7.688606e-03 ; 1.035267e-02 ];
Tc_error_9  = [ 8.108411e+00 ; 8.025227e+00 ; 7.170451e+00 ];

%-- Image #10:
omc_10 = [ 2.073790e+00 ; 2.020157e+00 ; -4.131868e-01 ];
Tc_10  = [ -3.171053e+02 ; -2.908214e+02 ; 1.336451e+03 ];
omc_error_10 = [ 5.439320e-03 ; 7.121627e-03 ; 1.229303e-02 ];
Tc_error_10  = [ 9.853496e+00 ; 9.858219e+00 ; 7.555606e+00 ];

%-- Image #11:
omc_11 = [ -1.853419e+00 ; -1.703015e+00 ; 1.164121e+00 ];
Tc_11  = [ 9.619946e+00 ; -2.351255e+02 ; 1.595818e+03 ];
omc_error_11 = [ 8.057811e-03 ; 4.197913e-03 ; 1.034935e-02 ];
Tc_error_11  = [ 1.183695e+01 ; 1.157995e+01 ; 6.868860e+00 ];

%-- Image #12:
omc_12 = [ 1.753588e+00 ; 1.796445e+00 ; 2.729295e-01 ];
Tc_12  = [ -7.661746e+01 ; -4.084646e+02 ; 1.101796e+03 ];
omc_error_12 = [ 6.232064e-03 ; 6.326222e-03 ; 9.897275e-03 ];
Tc_error_12  = [ 8.336498e+00 ; 8.068276e+00 ; 6.954087e+00 ];

%-- Image #13:
omc_13 = [ 1.772890e+00 ; 1.814058e+00 ; 2.465644e-01 ];
Tc_13  = [ -5.780351e+02 ; -1.940337e+02 ; 2.157532e+03 ];
omc_error_13 = [ 7.145003e-03 ; 8.323632e-03 ; 1.417399e-02 ];
Tc_error_13  = [ 1.606027e+01 ; 1.607463e+01 ; 1.437712e+01 ];

%-- Image #14:
omc_14 = [ -2.115229e+00 ; -2.015948e+00 ; 8.441023e-01 ];
Tc_14  = [ -5.256095e+01 ; -7.712659e+01 ; 2.426400e+03 ];
omc_error_14 = [ 9.375151e-03 ; 7.362158e-03 ; 1.602176e-02 ];
Tc_error_14  = [ 1.786043e+01 ; 1.773641e+01 ; 1.232710e+01 ];

%-- Image #15:
omc_15 = [ -1.948870e+00 ; -2.078699e+00 ; -1.013113e-01 ];
Tc_15  = [ 1.705479e+02 ; -3.089360e+02 ; 2.072645e+03 ];
omc_error_15 = [ 9.901765e-03 ; 1.366416e-02 ; 2.171758e-02 ];
Tc_error_15  = [ 1.518277e+01 ; 1.526966e+01 ; 1.355890e+01 ];

%-- Image #16:
omc_16 = [ 2.201461e+00 ; 2.120663e+00 ; 6.212504e-01 ];
Tc_16  = [ -1.379912e+02 ; -3.431460e+02 ; 2.114125e+03 ];
omc_error_16 = [ 9.301250e-03 ; 7.934965e-03 ; 1.713791e-02 ];
Tc_error_16  = [ 1.569730e+01 ; 1.558700e+01 ; 1.441022e+01 ];

