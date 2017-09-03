function ABCDEF = conic_mat2poly(C_conic)
% Input: x^T * C_conic * x = 0.
%	C_conic = [
%		A B/2 D/2
%		B/2 C E/2
%		D/2 E/2 F
%	]
% Output: A*x^2 + B*x*y + C*y^2 + D*x + E*y + F = 0.
% REF [site] >> https://en.wikipedia.org/wiki/Conic_section

ABCDEF = [C_conic(1,1), 2*C_conic(1,2), C_conic(2,2), 2*C_conic(1,3), 2*C_conic(2,3), C_conic(3,3)];
%if abs(ABCDEF(1)) > eps
%	ABCDEF = ABCDEF / ABCDEF(1);
%elseif abs(ABCDEF(3)) > eps
%	ABCDEF = ABCDEF / ABCDEF(3);
%end;
%[A, B, C, D, E, F] = deal(C_conic(1,1), 2*C_conic(1,2), C_conic(2,2), 2*C_conic(1,3), 2*C_conic(2,3), C_conic(3,3));
