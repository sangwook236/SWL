function coeffs = quadric_mat2poly(Q)
% Input: x^T * Q * x = 0.
%	Q = [
%		Q11 Q12/2 Q13/2 Q14/2
%		Q12/2 Q22 Q23/2 Q24/2
%		Q13/2 Q23/2 Q33 Q34/2
%		Q14/2 Q24/2 Q34/2 Q44
%	]
% Output: coeffs = [ Q11 Q22 Q33 Q12 Q23 Q13 Q14 Q24 Q34 Q44 ].
%	Q11*x^2 + Q22*y^2 + Q33*z^2 + Q12*x*y + Q23*y*z + Q13*z*x + Q14*x + Q24*y + Q34*z + Q44 = 0.
% REF [site] >> https://en.wikipedia.org/wiki/Quadric

coeffs = [Q(1,1), Q(2,2), Q(3,3), 2*Q(1,2), 2*Q(2,3), 2*Q(1,3), 2*Q(1,4), 2*Q(2,4), 2*Q(3,4), Q(4,4)];
%if abs(coeffs(1)) > eps
%	coeffs = coeffs / coeffs(1);
%elseif abs(coeffs(2)) > eps
%	coeffs = coeffs / coeffs(2);
%elseif abs(coeffs(3)) > eps
%	coeffs = coeffs / coeffs(3);
%end;
