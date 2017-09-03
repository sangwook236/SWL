function C_conic = conic_mat2poly(ABCDEF)
% Input: A*x^2 + B*x*y + C*y^2 + D*x + E*y + F = 0.
% Output: x^T * C_conic * x = 0.
%	C_conic = [
%		A B/2 D/2
%		B/2 C E/2
%		D/2 E/2 F
%	]
% REF [site] >> https://en.wikipedia.org/wiki/Conic_section

%C_conic = [
%	ABCDEF(1) ABCDEF(2)/2 ABCDEF(4)/2
%	ABCDEF(2)/2 ABCDEF(3) ABCDEF(5)/2
%	ABCDEF(4)/2 ABCDEF(5)/2 ABCDEF(6)
%];
C_conic = [
	2*ABCDEF(1) ABCDEF(2) ABCDEF(4)
	ABCDEF(2) 2*ABCDEF(3) ABCDEF(5)
	ABCDEF(4) ABCDEF(5) 2*ABCDEF(6)
];
