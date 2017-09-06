function [Ls, Linf, II, JJ, alpha, beta] = find_invariants_for_single_axis_geometry(ABCDEF)
% Intersection points of N imaged circles = the imaged circular points I & J at infinity.
%	Images of N circles in world planes = N ellipses in an image plane.
% ABCDEF: coefficients of conic(ellipse) equations. [N, 6].
%	A1*x^2 + B1*x*y + C1*y^2 + D1*x + E1*y + F1 = 0.
%	...
%	An*x^2 + Bn*x*y + Cn*y^2 + Dn*x + En*y + Fn = 0.
%
% Ls: the imaged rotation axis.
% Linf: the imaged line at infinity (vanishing line).
% II, JJ: the imaged circular points at infinity.
% alpha, beta: the circular points at infinity.
%	The coordinates [1 ; +-i ; 0] of the circular points on the metric plane to [alpha -+ beta*i ; 1 ; 0] on the affine plane.
% 
% REF [paper] >> "Single Axis Geometry by Fitting Conics", ECCV 2002.
% REF [book] >> "Multiple View Geometry in Computer Vision", p.490~.

%tol = eps * 1e5;
tol = 1.0e-10;

num_conics = size(ABCDEF, 1);

%----------
% Compute the imaged circular points I & J and the imaged line at infinity (vanishing line) Linf.

%[Linf, II, JJ, alpha, beta] = find_invariants_for_single_axis_geometry1(ABCDEF, tol);
[Linf, II, JJ, alpha, beta] = find_invariants_for_single_axis_geometry2(ABCDEF, tol);

%----------
% Compute the projections of circle centers = the poles of their imaged conics wrt the vanishing line.

OO = zeros(size(Linf,1), num_conics);
for ii = 1:num_conics
	Ci = conic_poly2mat(ABCDEF(ii,:));
	OO(:,ii) = inv(Ci) * Linf;
	OO(:,ii) = OO(:,ii) / OO(3,ii);
end;

%----------
% Compute the rotation axis.

%Ls = cross(OO(:,ii), OO(:,jj));  % FIXME [fix] >>
%Ls = linear_regression(OO(:,1)', OO(:,2)')'
Ls = orthogonal_linear_regression(OO(:,1)', OO(:,2)')';
Ls = Ls / Ls(3);

return;

%-----------------------------------------------------------

function [Linf, II, JJ, alpha, beta] = find_invariants_for_single_axis_geometry1(ABCDEF, tol)
% ABCDEF: coefficients of conic(ellipse) equations. [N, 6].
%	A1*x^2 + B1*x*y + C1*y^2 + D1*x + E1*y + F1 = 0.
%	...
%	An*x^2 + Bn*x*y + Cn*y^2 + Dn*x + En*y + Fn = 0.

% FIXME [fix] >>

for kk = 1:num_conics
	for ii = (kk+1):num_conics
		intsec_pts = find_intersections_of_two_conics(ABCDEF(kk,:), ABCDEF(ii,:))
	end;
end;

Linf = [];
II = [];
JJ = [];
alpha = [];
beta = [];

return;

%-----------------------------------------------------------

function [Linf, II, JJ, alpha, beta] = find_invariants_for_single_axis_geometry2(ABCDEF, tol)
% ABCDEF: coefficients of conic(ellipse) equations. [N, 6].
%	A1*x^2 + B1*x*y + C1*y^2 + D1*x + E1*y + F1 = 0.
%	...
%	An*x^2 + Bn*x*y + Cn*y^2 + Dn*x + En*y + Fn = 0.

x0 = rand([4, 1]);  % l1, l2, alpha, beta.

%options = optimoptions(@fminunc, 'Algorithm', 'quasi-newton', 'MaxIterations', 1000, 'MaxFunctionEvaluations', 1000, 'StepTolerance', 1e-6, 'OptimalityTolerance', 1e-6);
%[x_sol, fval, exitflag, output] = fminunc(@local_single_axis_geometry_objective, x0, options, ABCDEF);
options = optimset('MaxIter', 1000, 'TolX', 1e-6, 'TolFun', 1e-6);
[x_sol, fval, exitflag, output] = fminsearch(@local_single_axis_geometry_objective, x0, options, ABCDEF);
%options = optimoptions(@lsqnonlin, 'Algorithm', 'levenberg-marquardt', 'MaxIterations', 10000, 'MaxFunctionEvaluations', 10000, 'StepTolerance', 1e-6, 'OptimalityTolerance', 1e-6);
%[x_sol, resnorm, residual, exitflag, output] = lsqnonlin(@local_single_axis_geometry_objective, x0, [], [], options, ABCDEF);

l1 = x_sol(1);
l2 = x_sol(2);
l3 = 1;
alpha = x_sol(3);
beta = x_sol(4);

Linf = [l1 ; l2 ; l3];

if false
	ee = l2^2 + 2 * alpha * l1 * l2 + (beta^2 + alpha^2) * l1^2;
	aa = -((alpha^2 + beta^2) * l1 + alpha * l2);
	bb = -beta * l2;
	cc = -(alpha * l1 + l2);
	dd = beta * l1;

	II = [aa - bb*i ; cc - dd*i ; ee];
	JJ = [aa + bb*i ; cc + dd*i ; ee];
else
	ee = l2^2 + 2 * alpha * l1 * l2 + (beta^2 + alpha^2) * l1^2;
	aa = -((alpha^2 + beta^2) * l1 + alpha * l2) / ee;
	bb = -beta * l2 / ee;
	cc = -(alpha * l1 + l2) / ee;
	dd = beta * l1 / ee;

	II = [aa - bb*i ; cc - dd*i ; 1];
	JJ = [aa + bb*i ; cc + dd*i ; 1];
end;

return;

%-----------------------------------------------------------

function cost = local_single_axis_geometry_objective(x, ABCDEF)
% ABCDEF: coefficients of conic(ellipse) equations. [N, 6].
%	A1*x^2 + B1*x*y + C1*y^2 + D1*x + E1*y + F1 = 0.
%	...
%	An*x^2 + Bn*x*y + Cn*y^2 + Dn*x + En*y + Fn = 0.

% When the imaged line at infinity Linf = [l1 ; l2 ; l3],
% the imaged circular points I, J = [alpha -+ beta * i ; 1 ; -(l2+alpha*l1)/l3 +- beta*l1/l3*i].

l1 = x(1);
l2 = x(2);
l3 = 1;
alpha = x(3);
beta = x(4);

num_conics = size(ABCDEF, 1);
num_polys = nchoosek(num_conics, 2);

ee = l2^2 + 2 * alpha * l1 * l2 + (beta^2 + alpha^2) * l1^2;
aa = -((alpha^2 + beta^2) * l1 + alpha * l2) / ee;
bb = -beta * l2 / ee;
cc = -(alpha * l1 + l2) / ee;
dd = beta * l1 / ee;

II = [aa - bb*i ; cc - dd*i ; 1];
JJ = [aa + bb*i ; cc + dd*i ; 1];

cost = 0;
intersection_point_computation_method = 2;
if 1 == intersection_point_computation_method
	% Find common points of intersection points of pairs of conics.

	for ii = 1:num_conics
		for jj = (ii+1):num_conics
			A1 = ABCDEF(ii,1);
			B1 = ABCDEF(ii,2);
			C1 = ABCDEF(ii,3);
			D1 = ABCDEF(ii,4);
			E1 = ABCDEF(ii,5);
			F1 = ABCDEF(ii,6);

			A2 = ABCDEF(jj,1);
			B2 = ABCDEF(jj,2);
			C2 = ABCDEF(jj,3);
			D2 = ABCDEF(jj,4);
			E2 = ABCDEF(jj,5);
			F2 = ABCDEF(jj,6);

			g_ij = A1^2*C2^2*bb^4-2*A1*A2*C1*C2*bb^4-A1*B1*B2*C2*bb^4+A2*B1^2*C2*bb^4+A2^2*C1^2*bb^4+A1*B2^2*C1*bb^4-A2*B1*B2*C1*bb^4-6*A1^2*C2^2*aa^2*bb^2+12*A1*A2*C1*C2*aa^2*bb^2+6*A1*B1*B2*C2*aa^2*bb^2-6*A2*B1^2*C2*aa^2*bb^2-6*A2^2*C1^2*aa^2*bb^2-6*A1*B2^2*C1*aa^2*bb^2+6*A2*B1*B2*C1*aa^2*bb^2+3*A1*B1*C2*E2*aa*bb^2-6*A1*B2*C1*E2*aa*bb^2+3*A2*B1*C1*E2*aa*bb^2+3*A1*B2*C2*E1*aa*bb^2-6*A2*B1*C2*E1*aa*bb^2+3*A2*B2*C1*E1*aa*bb^2+6*A1*C1*C2*D2*aa*bb^2-3*B1^2*C2*D2*aa*bb^2-6*A2*C1^2*D2*aa*bb^2+3*B1*B2*C1*D2*aa*bb^2-6*A1*C2^2*D1*aa*bb^2+6*A2*C1*C2*D1*aa*bb^2+3*B1*B2*C2*D1*aa*bb^2-3*B2^2*C1*D1*aa*bb^2+2*A1*C1*C2*F2*bb^2-B1^2*C2*F2*bb^2-2*A2*C1^2*F2*bb^2+B1*B2*C1*F2*bb^2-2*A1*C2^2*F1*bb^2+2*A2*C1*C2*F1*bb^2+B1*B2*C2*F1*bb^2-B2^2*C1*F1*bb^2-A1*C1*E2^2*bb^2+A1*C2*E1*E2*bb^2+A2*C1*E1*E2*bb^2+B1*C1*D2*E2*bb^2+B1*C2*D1*E2*bb^2-2*B2*C1*D1*E2*bb^2-A2*C2*E1^2*bb^2-2*B1*C2*D2*E1*bb^2+B2*C1*D2*E1*bb^2+B2*C2*D1*E1*bb^2-C1^2*D2^2*bb^2+2*C1*C2*D1*D2*bb^2-C2^2*D1^2*bb^2+A1^2*C2^2*aa^4-2*A1*A2*C1*C2*aa^4-A1*B1*B2*C2*aa^4+A2*B1^2*C2*aa^4+A2^2*C1^2*aa^4+A1*B2^2*C1*aa^4-A2*B1*B2*C1*aa^4-A1*B1*C2*E2*aa^3+2*A1*B2*C1*E2*aa^3-A2*B1*C1*E2*aa^3-A1*B2*C2*E1*aa^3+2*A2*B1*C2*E1*aa^3-A2*B2*C1*E1*aa^3-2*A1*C1*C2*D2*aa^3+B1^2*C2*D2*aa^3+2*A2*C1^2*D2*aa^3-B1*B2*C1*D2*aa^3+2*A1*C2^2*D1*aa^3-2*A2*C1*C2*D1*aa^3-B1*B2*C2*D1*aa^3+B2^2*C1*D1*aa^3-2*A1*C1*C2*F2*aa^2+B1^2*C2*F2*aa^2+2*A2*C1^2*F2*aa^2-B1*B2*C1*F2*aa^2+2*A1*C2^2*F1*aa^2-2*A2*C1*C2*F1*aa^2-B1*B2*C2*F1*aa^2+B2^2*C1*F1*aa^2+A1*C1*E2^2*aa^2-A1*C2*E1*E2*aa^2-A2*C1*E1*E2*aa^2-B1*C1*D2*E2*aa^2-B1*C2*D1*E2*aa^2+2*B2*C1*D1*E2*aa^2+A2*C2*E1^2*aa^2+2*B1*C2*D2*E1*aa^2-B2*C1*D2*E1*aa^2-B2*C2*D1*E1*aa^2+C1^2*D2^2*aa^2-2*C1*C2*D1*D2*aa^2+C2^2*D1^2*aa^2-B1*C1*E2*F2*aa+2*B1*C2*E1*F2*aa-B2*C1*E1*F2*aa+2*C1^2*D2*F2*aa-2*C1*C2*D1*F2*aa-B1*C2*E2*F1*aa+2*B2*C1*E2*F1*aa-B2*C2*E1*F1*aa-2*C1*C2*D2*F1*aa+2*C2^2*D1*F1*aa+C1*D1*E2^2*aa-C1*D2*E1*E2*aa-C2*D1*E1*E2*aa+C2*D2*E1^2*aa+C1^2*F2^2-2*C1*C2*F1*F2-C1*E1*E2*F2+C2*E1^2*F2+C2^2*F1^2+C1*E2^2*F1-C2*E1*E2*F1;
			h_ij = -4*A1^2*C2^2*aa*bb^3+8*A1*A2*C1*C2*aa*bb^3+4*A1*B1*B2*C2*aa*bb^3-4*A2*B1^2*C2*aa*bb^3-4*A2^2*C1^2*aa*bb^3-4*A1*B2^2*C1*aa*bb^3+4*A2*B1*B2*C1*aa*bb^3+A1*B1*C2*E2*bb^3-2*A1*B2*C1*E2*bb^3+A2*B1*C1*E2*bb^3+A1*B2*C2*E1*bb^3-2*A2*B1*C2*E1*bb^3+A2*B2*C1*E1*bb^3+2*A1*C1*C2*D2*bb^3-B1^2*C2*D2*bb^3-2*A2*C1^2*D2*bb^3+B1*B2*C1*D2*bb^3-2*A1*C2^2*D1*bb^3+2*A2*C1*C2*D1*bb^3+B1*B2*C2*D1*bb^3-B2^2*C1*D1*bb^3+4*A1^2*C2^2*aa^3*bb-8*A1*A2*C1*C2*aa^3*bb-4*A1*B1*B2*C2*aa^3*bb+4*A2*B1^2*C2*aa^3*bb+4*A2^2*C1^2*aa^3*bb+4*A1*B2^2*C1*aa^3*bb-4*A2*B1*B2*C1*aa^3*bb-3*A1*B1*C2*E2*aa^2*bb+6*A1*B2*C1*E2*aa^2*bb-3*A2*B1*C1*E2*aa^2*bb-3*A1*B2*C2*E1*aa^2*bb+6*A2*B1*C2*E1*aa^2*bb-3*A2*B2*C1*E1*aa^2*bb-6*A1*C1*C2*D2*aa^2*bb+3*B1^2*C2*D2*aa^2*bb+6*A2*C1^2*D2*aa^2*bb-3*B1*B2*C1*D2*aa^2*bb+6*A1*C2^2*D1*aa^2*bb-6*A2*C1*C2*D1*aa^2*bb-3*B1*B2*C2*D1*aa^2*bb+3*B2^2*C1*D1*aa^2*bb-4*A1*C1*C2*F2*aa*bb+2*B1^2*C2*F2*aa*bb+4*A2*C1^2*F2*aa*bb-2*B1*B2*C1*F2*aa*bb+4*A1*C2^2*F1*aa*bb-4*A2*C1*C2*F1*aa*bb-2*B1*B2*C2*F1*aa*bb+2*B2^2*C1*F1*aa*bb+2*A1*C1*E2^2*aa*bb-2*A1*C2*E1*E2*aa*bb-2*A2*C1*E1*E2*aa*bb-2*B1*C1*D2*E2*aa*bb-2*B1*C2*D1*E2*aa*bb+4*B2*C1*D1*E2*aa*bb+2*A2*C2*E1^2*aa*bb+4*B1*C2*D2*E1*aa*bb-2*B2*C1*D2*E1*aa*bb-2*B2*C2*D1*E1*aa*bb+2*C1^2*D2^2*aa*bb-4*C1*C2*D1*D2*aa*bb+2*C2^2*D1^2*aa*bb-B1*C1*E2*F2*bb+2*B1*C2*E1*F2*bb-B2*C1*E1*F2*bb+2*C1^2*D2*F2*bb-2*C1*C2*D1*F2*bb-B1*C2*E2*F1*bb+2*B2*C1*E2*F1*bb-B2*C2*E1*F1*bb-2*C1*C2*D2*F1*bb+2*C2^2*D1*F1*bb+C1*D1*E2^2*bb-C1*D2*E1*E2*bb-C2*D1*E1*E2*bb+C2*D2*E1^2*bb;

			cost = cost + g_ij^2 + h_ij^2;
		end;
	end;
elseif 2 == intersection_point_computation_method
	% Find common intersection points of all conics.

	for ii = 1:num_conics
		CC = conic_poly2mat(ABCDEF(ii,:));
		cost = cost + abs(transpose(II) * CC * II)^2 + abs(transpose(JJ) * CC * JJ)^2;
		%CI = CC * II;
		%CJ = CC * JJ;
		%cost = cost + 0.25 * ((transpose(II) * CI)^2 / (CI(1)^2 + CI(2)^2) + (transpose(JJ) * CJ)^2 / (CJ(1)^2 + CJ(2)^2));
	end;
else
	error('[Error] Invalid intersection point computation method.');
end;

cost = sqrt(cost / num_polys);  % RMS.

return;
