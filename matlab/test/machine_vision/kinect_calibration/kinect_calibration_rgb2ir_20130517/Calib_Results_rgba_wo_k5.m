% Intrinsic and Extrinsic Camera Parameters
%
% This script file can be directly excecuted under Matlab to recover the camera intrinsic and extrinsic parameters.
% IMPORTANT: This file contains neither the structure of the calibration objects nor the image coordinates of the calibration points.
%            All those complementary variables are saved in the complete matlab data file Calib_Results.mat.
% For more information regarding the calibration model visit http://www.vision.caltech.edu/bouguetj/calib_doc/


%-- Focal length:
fc = [ 526.681423129443710 ; 528.064146617164280 ];

%-- Principal point:
cc = [ 327.652895418469710 ; 265.205963685449150 ];

%-- Skew coefficient:
alpha_c = 0.000000000000000;

%-- Distortion coefficients:
kc = [ 0.232225515185403 ; -0.559813783976062 ; 0.002277053552942 ; 0.003720963676783 ; 0.000000000000000 ];

%-- Focal length uncertainty:
fc_error = [ 2.582068243378091 ; 2.473426976206287 ];

%-- Principal point uncertainty:
cc_error = [ 3.352669081944522 ; 3.442906013171974 ];

%-- Skew coefficient uncertainty:
alpha_c_error = 0.000000000000000;

%-- Distortion coefficients uncertainty:
kc_error = [ 0.019328627079994 ; 0.093922517789420 ; 0.003273959993899 ; 0.002949057675206 ; 0.000000000000000 ];

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
omc_1 = [ -1.971765e+00 ; -1.993895e+00 ; 1.225617e-01 ];
Tc_1  = [ -1.962367e+02 ; -2.323327e+02 ; 1.484400e+03 ];
omc_error_1 = [ 5.927312e-03 ; 6.286975e-03 ; 1.237932e-02 ];
Tc_error_1  = [ 9.442920e+00 ; 9.683068e+00 ; 7.314194e+00 ];

%-- Image #2:
omc_2 = [ 2.118124e+00 ; 1.815305e+00 ; 7.221938e-01 ];
Tc_2  = [ -4.713628e+02 ; -2.455384e+02 ; 1.537272e+03 ];
omc_error_2 = [ 6.932694e-03 ; 5.557140e-03 ; 1.181329e-02 ];
Tc_error_2  = [ 1.009540e+01 ; 1.030149e+01 ; 9.359005e+00 ];

%-- Image #3:
omc_3 = [ -1.991713e+00 ; -1.915972e+00 ; -3.250359e-01 ];
Tc_3  = [ -1.514242e+02 ; -5.366727e+02 ; 1.826003e+03 ];
omc_error_3 = [ 8.274772e-03 ; 8.613977e-03 ; 1.560920e-02 ];
Tc_error_3  = [ 1.188195e+01 ; 1.231118e+01 ; 1.045437e+01 ];

%-- Image #4:
omc_4 = [ 2.126030e+00 ; 1.527723e+00 ; 1.126057e+00 ];
Tc_4  = [ -5.138853e+02 ; -1.820513e+02 ; 1.272012e+03 ];
omc_error_4 = [ 6.952196e-03 ; 4.133453e-03 ; 9.194256e-03 ];
Tc_error_4  = [ 8.576314e+00 ; 8.726576e+00 ; 8.336153e+00 ];

%-- Image #5:
omc_5 = [ -1.562996e+00 ; -1.800914e+00 ; 2.443201e-01 ];
Tc_5  = [ -3.939737e+02 ; -2.662977e+02 ; 1.431618e+03 ];
omc_error_5 = [ 5.754451e-03 ; 5.276791e-03 ; 8.192513e-03 ];
Tc_error_5  = [ 9.152835e+00 ; 9.471165e+00 ; 7.025769e+00 ];

%-- Image #6:
omc_6 = [ -2.007711e+00 ; -1.852459e+00 ; -4.287087e-01 ];
Tc_6  = [ -2.258687e+02 ; -1.370125e+02 ; 9.963502e+02 ];
omc_error_6 = [ 4.363666e-03 ; 5.861453e-03 ; 9.356905e-03 ];
Tc_error_6  = [ 6.364115e+00 ; 6.539654e+00 ; 5.407955e+00 ];

%-- Image #7:
omc_7 = [ -1.539648e+00 ; -1.784640e+00 ; -2.592986e-01 ];
Tc_7  = [ -4.902792e+02 ; -2.341225e+02 ; 1.277649e+03 ];
omc_error_7 = [ 5.429074e-03 ; 5.610491e-03 ; 8.111636e-03 ];
Tc_error_7  = [ 8.171698e+00 ; 8.636497e+00 ; 7.625104e+00 ];

%-- Image #8:
omc_8 = [ -2.084451e+00 ; -1.688522e+00 ; -1.018666e+00 ];
Tc_8  = [ -4.531901e+02 ; -8.502421e+01 ; 1.109694e+03 ];
omc_error_8 = [ 5.212341e-03 ; 6.067206e-03 ; 9.124699e-03 ];
Tc_error_8  = [ 7.153511e+00 ; 7.568939e+00 ; 7.269483e+00 ];

%-- Image #9:
omc_9 = [ -2.074414e+00 ; -1.649114e+00 ; -1.096782e+00 ];
Tc_9  = [ -1.258929e+02 ; -1.266301e+02 ; 1.094240e+03 ];
omc_error_9 = [ 4.212337e-03 ; 6.805672e-03 ; 9.066933e-03 ];
Tc_error_9  = [ 7.033092e+00 ; 7.154850e+00 ; 6.328981e+00 ];

%-- Image #10:
omc_10 = [ 2.074564e+00 ; 2.022351e+00 ; -4.147327e-01 ];
Tc_10  = [ -3.136297e+02 ; -2.911793e+02 ; 1.337128e+03 ];
omc_error_10 = [ 4.838305e-03 ; 6.276939e-03 ; 1.075235e-02 ];
Tc_error_10  = [ 8.538529e+00 ; 8.783410e+00 ; 6.656013e+00 ];

%-- Image #11:
omc_11 = [ -1.850703e+00 ; -1.701306e+00 ; 1.165913e+00 ];
Tc_11  = [ 1.374779e+01 ; -2.352863e+02 ; 1.595738e+03 ];
omc_error_11 = [ 7.055324e-03 ; 3.670408e-03 ; 9.100599e-03 ];
Tc_error_11  = [ 1.025124e+01 ; 1.031643e+01 ; 6.048157e+00 ];

%-- Image #12:
omc_12 = [ 1.754583e+00 ; 1.798364e+00 ; 2.740818e-01 ];
Tc_12  = [ -7.353521e+01 ; -4.087812e+02 ; 1.101608e+03 ];
omc_error_12 = [ 5.517235e-03 ; 5.459436e-03 ; 8.731642e-03 ];
Tc_error_12  = [ 7.219974e+00 ; 7.190911e+00 ; 6.130621e+00 ];

%-- Image #13:
omc_13 = [ 1.773449e+00 ; 1.815962e+00 ; 2.457428e-01 ];
Tc_13  = [ -5.719895e+02 ; -1.943387e+02 ; 2.156730e+03 ];
omc_error_13 = [ 6.293188e-03 ; 7.268345e-03 ; 1.246625e-02 ];
Tc_error_13  = [ 1.391473e+01 ; 1.430868e+01 ; 1.265754e+01 ];

%-- Image #14:
omc_14 = [ -2.111522e+00 ; -2.013012e+00 ; 8.464315e-01 ];
Tc_14  = [ -4.625125e+01 ; -7.735458e+01 ; 2.426240e+03 ];
omc_error_14 = [ 8.197162e-03 ; 6.458405e-03 ; 1.398199e-02 ];
Tc_error_14  = [ 1.545758e+01 ; 1.579369e+01 ; 1.085567e+01 ];

%-- Image #15:
omc_15 = [ -1.948342e+00 ; -2.077749e+00 ; -9.584932e-02 ];
Tc_15  = [ 1.757246e+02 ; -3.093321e+02 ; 2.072691e+03 ];
omc_error_15 = [ 8.652005e-03 ; 1.191547e-02 ; 1.929069e-02 ];
Tc_error_15  = [ 1.316182e+01 ; 1.361241e+01 ; 1.195843e+01 ];

%-- Image #16:
omc_16 = [ 2.202227e+00 ; 2.120428e+00 ; 6.217027e-01 ];
Tc_16  = [ -1.323932e+02 ; -3.432079e+02 ; 2.112087e+03 ];
omc_error_16 = [ 8.147018e-03 ; 6.938048e-03 ; 1.508674e-02 ];
Tc_error_16  = [ 1.357994e+01 ; 1.386046e+01 ; 1.268063e+01 ];

