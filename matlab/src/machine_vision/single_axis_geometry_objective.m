function cost = single_axis_geometry_objective(x, pt)
% x = [1l, l2, alpha, beta, ls_x, ls_y, t1, r1, t2, r2, ...].
%
% REF [paper] >> "Signle Axis Geometry by Fitting Conics", ECCV 2002.
% REF [book] >> "Multiple View Geometry in Computer Vision", p.490~.

num_conics = length(pt);

l1 = x(1);
l2 = x(2);
l3 = 1;
alpha = x(3);
beta = x(4);
ls_x = x(5);
ls_y = x(6);
t = zeros(1, num_conics);
r = zeros(1, num_conics);
idx = 7;
for kk = 1:num_conics
	t(kk) = x(idx);
	r(kk) = x(idx + 1);
	idx = idx + 2;
end;

if abs(beta) < eps
	disp('[Warning] beta is nearly zero');
	beta = eps * 100;
end;

%P = [
%	1 0 0
%	0 1 0
%	l1 l2 l3
%];
%A = [
%	1/beta -alpha/beta 0
%	0 1 0
%	0 0 1
%];
AP = [
	1/beta -alpha/beta 0
	0 1 0
	l1 l2 l3
];
%R = eye(3);

C_circle = [
	1 0 0
	0 1 0
	0 0 -1
];

cost = 0;
for kk = 1:num_conics
	t1 = t(kk);
	t2 = -((beta*l3*ls_x - beta*l1)*t1 + 1) / (l3*ls_y + alpha*l3*ls_x - l2 - alpha*l1);
	s = r(kk);

	%T = [
	%	1 0 -t1
	%	0 1 -t2
	%	0 0 1
	%];
	%S = [
	%	s 0 0
	%	0 s 0
	%	0 0 1
	%];
	%H = R * S * T * A * P;
	RST = [
		s 0 -s*t1
		0 s -s*t2
		0 0 1
	];
	H = RST * AP;
	C_conic = H' * C_circle * H;
	CX = C_conic * pt{kk}';

	len = size(pt{kk}, 1);
	for ii = 1:len
		cost = cost + 0.25 * (pt{kk}(ii,:) * CX(:,ii))^2 ./ (CX(1,ii)^2 + CX(2,ii)^2);
	end;
end;
