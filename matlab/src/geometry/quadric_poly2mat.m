function Q = quadric_mat2poly(coeffs)
% Input: coeffs = [ Q11 Q22 Q33 Q12 Q23 Q13 Q14 Q24 Q34 Q44 ].
%	Q11*x^2 + Q22*y^2 + Q33*z^2 + Q12*x*y + Q23*y*z + Q13*z*x + Q14*x + Q24*y + Q34*z + Q44 = 0.
% Output: x^T * Q * x = 0.
%	Q = [
%		Q11 Q12/2 Q13/2 Q14/2
%		Q12/2 Q22 Q23/2 Q24/2
%		Q13/2 Q23/2 Q33 Q34/2
%		Q14/2 Q24/2 Q34/2 Q44
%	]
% REF [site] >> https://en.wikipedia.org/wiki/Quadric

%Q = [
%	coeffs(1) coeffs(4)/2 coeffs(6)/2 coeffs(7)/2
%	coeffs(4)/2 coeffs(2) coeffs(5)/2 coeffs(8)/2
%	coeffs(6)/2 coeffs(5)/2 coeffs(3) coeffs(9)/2
%	coeffs(7)/2 coeffs(8)/2 coeffs(9)/2 coeffs(10)
%];
Q = [
	2*coeffs(1) coeffs(4) coeffs(6) coeffs(7)
	coeffs(4) 2*coeffs(2) coeffs(5) coeffs(8)
	coeffs(6) coeffs(5) 2*coeffs(3) coeffs(9)
	coeffs(7) coeffs(8) coeffs(9) 2*coeffs(10)
];
