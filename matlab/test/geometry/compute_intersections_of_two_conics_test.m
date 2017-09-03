%addpath('D:/work/SWL_github/matlab/src/geometry');

%-----------------------------------------------------------

coeffs1 = [1 0 4 0 8 -12];
coeffs2 = [6 0 1 0 4 -8];
sols1 = compute_intersections_of_two_conics(coeffs1, coeffs2)

coeffs1 = [1 0 1 0 0 -1];
coeffs2 = [1 0 1 -2 -2 1];
sols2 = compute_intersections_of_two_conics(coeffs1, coeffs2)
