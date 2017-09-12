function [Ip, Jp, a, b, c, d] = find_imaged_circular_points_for_single_axis_geometry(ABCDEF)
% Intersection points of N imaged circles = the imaged circular points I & J at infinity on the projective plane.
%	Images of N circles in world planes = N ellipses in an image plane.
% ABCDEF: coefficients of conic(ellipse) equations. [N, 6].
%	A1*x^2 + B1*x*y + C1*y^2 + D1*x + E1*y + F1 = 0.
%	...
%	An*x^2 + Bn*x*y + Cn*y^2 + Dn*x + En*y + Fn = 0.
%
% Ip, Jp: the imaged circular points at infinity on the projective plane.
%	Ip = [a-b*i ; c-d*i ; 1], Jp = [a+b*i ; c+d*i ; 1].
% 
% REF [paper] >> "Single Axis Geometry by Fitting Conics", ECCV 2002.
% REF [book] >> "Multiple View Geometry in Computer Vision", p.490~.

num_conics = size(ABCDEF, 1);

x0 = rand([4, 1]);  % [a, b, c, d].

%options = optimoptions(@fminunc, 'Algorithm', 'quasi-newton');
%%options = optimoptions(@fminunc, 'Algorithm', 'quasi-newton', 'MaxIterations', 1000, 'MaxFunctionEvaluations', 1000, 'StepTolerance', 1e-6, 'OptimalityTolerance', 1e-6);
%[x_sol, fval, exitflag, output] = fminunc(@local_single_axis_geometry_objective, x0, options, ABCDEF);
%options = optimset();
%%options = optimset('MaxIter', 1000, 'TolX', 1e-6, 'TolFun', 1e-6);
%[x_sol, fval, exitflag, output] = fminsearch(@local_single_axis_geometry_objective, x0, options, ABCDEF);
options = optimoptions(@lsqnonlin, 'Algorithm', 'levenberg-marquardt');
%options = optimoptions(@lsqnonlin, 'Algorithm', 'levenberg-marquardt', 'MaxIterations', 10000, 'MaxFunctionEvaluations', 10000, 'StepTolerance', 1e-6, 'OptimalityTolerance', 1e-6);
[x_sol, resnorm, residual, exitflag, output] = lsqnonlin(@local_single_axis_geometry_objective, x0, [], [], options, ABCDEF);

a = x_sol(1);
b = x_sol(2);
c = x_sol(3);
d = x_sol(4);

Ip = [a-b*i ; c-d*i ; 1];
Jp = [a+b*i ; c+d*i ; 1];

return;

%-----------------------------------------------------------

function cost = local_single_axis_geometry_objective(x, ABCDEF)

a = x(1);
b = x(2);
c = x(3);
d = x(4);

num_conics = size(ABCDEF, 1);
cost = 0;
for kk = 1:num_conics
	A1 = ABCDEF(kk,1);
	B1 = ABCDEF(kk,2);
	C1 = ABCDEF(kk,3);
	D1 = ABCDEF(kk,4);
	E1 = ABCDEF(kk,5);
	F1 = ABCDEF(kk,6);

	Gn = -C1*d^2-B1*b*d+C1*c^2+(B1*a+E1)*c-A1*b^2+A1*a^2+D1*a+F1;
	Hn = (-2*C1*c-B1*a-E1)*d-B1*b*c+(-2*A1*a-D1)*b;

	cost = cost + Gn^2 + Hn^2;
end;

return;
