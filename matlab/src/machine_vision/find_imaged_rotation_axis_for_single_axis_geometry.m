function [Ls, O] = find_imaged_rotation_axis_for_single_axis_geometry(ABCDEF, Linf)
% ABCDEF: coefficients of conic(ellipse) equations. [N, 6].
%	A1*x^2 + B1*x*y + C1*y^2 + D1*x + E1*y + F1 = 0.
%	...
%	An*x^2 + Bn*x*y + Cn*y^2 + Dn*x + En*y + Fn = 0.
% Ls: the imaged line at infinity (vanishing line) on the projective plane.
%
% Ls: the imaged rotation axis on the projective plane.
% 
% REF [paper] >> "Single Axis Geometry by Fitting Conics", ECCV 2002.
% REF [book] >> "Multiple View Geometry in Computer Vision", p.490~.

num_conics = size(ABCDEF, 1);

%----------
% Compute the projections of circle centers = the poles of their imaged conics wrt the vanishing line.

O = zeros(size(Linf,1), num_conics);
for kk = 1:num_conics
	O(:,kk) = inv(conic_poly2mat(ABCDEF(kk,:))) * Linf;
	O(:,kk) = O(:,kk) / O(3,kk);
end;

%----------
% Compute the rotation axis.

%Ls = cross(O(:,ii), O(:,jj));  % FIXME [fix] >>
%Ls = linear_regression(O(1,:)', O(2,:)')'
Ls = orthogonal_linear_regression(O(1,:)', O(2,:)')';
%Ls = Ls / Ls(3);
