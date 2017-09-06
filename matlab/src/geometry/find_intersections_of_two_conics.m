function sols = find_intersections_of_two_conics(ABCDEF1, ABCDEF2)
% A1*x^2 + B1*x*y + C1*y^2 + D1*x + E1*y + F1 = 0.
% A2*x^2 + B2*x*y + C2*y^2 + D2*x + E2*y + F2 = 0.

% NOTICE [important] >> Computing intersection points of two circles has to be specially treated as exceptional cases.

%tol = eps * 1e5;
tol = 1.0e-10;

sols = find_intersections_of_two_conics1(ABCDEF1, ABCDEF2, tol);  % Partial solutions in some speical cases.
%sols = find_intersections_of_two_conics2(ABCDEF1, ABCDEF2, tol);  % Not correctly working.
%sols = find_intersections_of_two_conics3(ABCDEF1, ABCDEF2, tol);  % Partial solutions if at least one of two conics is a circle.

return;

%-----------------------------------------------------------

function sols = find_intersections_of_two_conics1(ABCDEF1, ABCDEF2, tol)
% A1*x^2 + B1*x*y + C1*y^2 + D1*x + E1*y + F1 = 0.
% A2*x^2 + B2*x*y + C2*y^2 + D2*x + E2*y + F2 = 0.
% REF [site] >> https://elliotnoma.wordpress.com/2013/04/10/a-closed-form-solution-for-the-intersections-of-two-ellipses/

A1 = ABCDEF1(1);
B1 = ABCDEF1(2);
C1 = ABCDEF1(3);
D1 = ABCDEF1(4);
E1 = ABCDEF1(5);
F1 = ABCDEF1(6);

A2 = ABCDEF2(1);
B2 = ABCDEF2(2);
C2 = ABCDEF2(3);
D2 = ABCDEF2(4);
E2 = ABCDEF2(5);
F2 = ABCDEF2(6);

% a * x^4 + b * x^3 + c * x^2 + d * x + e = 0.
a = A1^2*C2^2+C1*(A1*(B2^2-2*A2*C2)-A2*B1*B2)-A1*B1*B2*C2+A2*B1^2*C2+A2^2*C1^2;
b = C1*(A1*(2*B2*E2-2*C2*D2)+B1*(-A2*E2-B2*D2))-A1*B1*C2*E2...
	+(-A1*B2*C2+2*A2*B1*C2-A2*B2*C1)*E1+B1^2*C2*D2+2*A2*C1^2*D2...
	+(2*A1*C2^2+C1*(B2^2-2*A2*C2)-B1*B2*C2)*D1;
c = C1*(A1*(E2^2-2*C2*F2)+B1*(-B2*F2-D2*E2))+C1^2*(2*A2*F2+D2^2)+B1^2*C2*F2...
	+(2*A1*C2^2+C1*(B2^2-2*A2*C2)-B1*B2*C2)*F1+D1*(C1*(2*B2*E2-2*C2*D2)-B1*C2*E2)...
	+E1*(C1*(-A2*E2-B2*D2)-A1*C2*E2+2*B1*C2*D2-B2*C2*D1)+A2*C2*E1^2+C2^2*D1^2;
d = E1*(C1*(-B2*F2-D2*E2)+2*B1*C2*F2-C2*D1*E2)+C1*D1*(E2^2-2*C2*F2)-B1*C1*E2*F2...
	+2*C1^2*D2*F2+(C1*(2*B2*E2-2*C2*D2)-B1*C2*E2-B2*C2*E1+2*C2^2*D1)*F1+C2*D2*E1^2;
e = C1^2*F2^2+F1*(C1*(E2^2-2*C2*F2)-C2*E1*E2)-C1*E1*E2*F2+C2*E1^2*F2+C2^2*F1^2;

x = roots([a b c d e]);

% a * y^2 + b * y + c = 0.
aa = C1;
bb = B1 * x + E1;
cc = A1 * x.^2 + D1 * x + F1;

if abs(aa) > tol
	determinant = sqrt(bb.^2 - 4 * aa * cc);
	yy = [(-bb + determinant) (-bb - determinant)] / (2 * aa);
	func1 = abs(A2 * x.^2 + B2 * x.*yy(:,1) + C2 * yy(:,1).^2 + D2 * x + E2 * yy(:,1) + F2);
	func2 = abs(A2 * x.^2 + B2 * x.*yy(:,2) + C2 * yy(:,2).^2 + D2 * x + E2 * yy(:,2) + F2);
	[X, I] = min([func1 func2], [], 2);
	y = zeros(size(x));
	for ii = 1:length(I)
		y(ii) = yy(ii,I(ii));
	end;
elseif all(abs(bb) > tol)
	y = -cc ./ bb;
else
	error('[Error] Improper quadratic equation.')
end;

sols = [x' ; y'];
%sols = [x' ; y' ; ones(size(x'))];

return;

%-----------------------------------------------------------

function sols = find_intersections_of_two_conics2(ABCDEF1, ABCDEF2, tol)
% A1*x^2 + B1*x*y + C1*y^2 + D1*x + E1*y + F1 = 0.
% A2*x^2 + B2*x*y + C2*y^2 + D2*x + E2*y + F2 = 0.
% REF [site] >> https://elliotnoma.wordpress.com/2013/04/10/a-closed-form-solution-for-the-intersections-of-two-ellipses/

A1 = ABCDEF1(1);
B1 = ABCDEF1(2);
C1 = ABCDEF1(3);
D1 = ABCDEF1(4);
E1 = ABCDEF1(5);
F1 = ABCDEF1(6);

A2 = ABCDEF2(1);
B2 = ABCDEF2(2);
C2 = ABCDEF2(3);
D2 = ABCDEF2(4);
E2 = ABCDEF2(5);
F2 = ABCDEF2(6);

% a * y^4 + b * y^3 + c * y^2 + d * y + e = 0.
a = A1^2*C2^2-2*A1*C2*A2*C1+A2^2*C1^2-B1*A1*B2*C2-B1*B2*A2*C1+B1^2*A2*C2+C1*A1*B2^2;
b = -2*A1*A2*C1*E2+E2*A2*B1^2+2*C2*B1*A2*D1-C1*A2*B2*D1+B2^2*A1*E1-E2*B2*A1*B1-2*A1*C2*A2*E1...
	-E1*A2*B2*B1-C2*B2*A1*D1+2*E2*C2*A1^2+2*E1*C1*A2^2-C1*A2*D2*B1+2*D2*B2*A1*C1-C2*D2*A1*B1;
c = E2^2*A1^2+2*C2*F2*A1^2-E1*A2*D2*B1+F2*A2*B1^2-E1*A2*B2*D1-F2*B2*A1*B1-2*A1*E2*A2*E1...
	+2*D2*B2*A1*E1-C2*D2*A1*D1-2*A1*C2*A2*F1+B2^2*A1*F1+2*E2*B1*A2*D1+E1^2*A2^2-C1*A2*D2*D1...
	-E2*B2*A1*D1+2*F1*C1*A2^2-F1*A2*B2*B1+C2*D1^2*A2+D2^2*A1*C1-E2*D2*A1*B1-2*A1*F2*A2*C1;
d = E2*D1^2*A2-F2*D2*A1*B1-2*A1*F2*A2*E1-F1*A2*B2*D1+2*D2*B2*A1*F1+2*E2*F2*A1^2+D2^2*A1*E1...
	-E2*D2*A1*D1-2*A1*E2*A2*F1-F1*A2*D2*B1+2*F1*E1*A2^2-F2*B2*A1*D1-E1*A2*D2*D1+2*F2*B1*A2*D1;
e = F1*A1*D2^2+A1^2*F2^2-D1*A1*D2*F2+A2^2*F1^2-2*A1*F2*A2*F1-D1*D2*A2*F1+A2*D1^2*F2;

y = roots([a b c d e]);

denom = (A1-A2)*(B1*y+D1);
if abs(denom) > tol
	x = -((A1*C2-A2*C1)*y.^2+(A1*E2-A2*E1)*y+A1*F2-A2*F1) ./ denom;

	sols = [x' ; y'];
	%sols = [x' ; y' ; ones(size(x'))];
else
	% Cases where the denominator is zero when the main axes of the ellipses are horizontal and vertical.
	%	There are only two different values out of 4 y values.

	yy = find_unique(y, 1.0e-5);

	% FIXME [fix] >> There are some errors.

	bb = B1 * yy + D1;
	cc = C1 * yy.^2 + E1 * yy + F1;
	sqrt_discriminant = sqrt(bb.^2 - 4 * A1 * cc);
	x1 = (-bb + sqrt_discriminant) / (2 * A1);
	x2 = (-bb - sqrt_discriminant) / (2 * A1);

	sols = [x1 x2 ; yy yy];
	%sols = [x1 x2 ; yy yy ; ones(size([x1 x2]))];
end;

return;

%-----------------------------------------------------------

function sols = find_intersections_of_two_conics3(ABCDEF1, ABCDEF2, tol)
% A1*x^2 + B1*x*y + C1*y^2 + D1*x + E1*y + F1 = 0.
% A2*x^2 + B2*x*y + C2*y^2 + D2*x + E2*y + F2 = 0.
% REF [site] >> https://en.wikipedia.org/wiki/Conic_section#Intersecting_two_conics
% REF [site] >> https://math.stackexchange.com/questions/425366/finding-intersection-of-an-ellipse-with-another-ellipse-when-both-are-rotated/425412#425412
% REF [site] >> https://math.stackexchange.com/questions/2332007/find-intersection-of-hyperbola-and-ellipse
% Bezout's theorem.
%	REF [site] >> https://en.wikipedia.org/wiki/B%C3%A9zout%27s_theorem

A1 = ABCDEF1(1);
B1 = ABCDEF1(2);
C1 = ABCDEF1(3);
D1 = ABCDEF1(4);
E1 = ABCDEF1(5);
F1 = ABCDEF1(6);

A2 = ABCDEF2(1);
B2 = ABCDEF2(2);
C2 = ABCDEF2(3);
D2 = ABCDEF2(4);
E2 = ABCDEF2(5);
F2 = ABCDEF2(6);

Cc1 = conic_poly2mat(ABCDEF1);
Cc2 = conic_poly2mat(ABCDEF2);

rankCc1 = rank(Cc1);
rankCc2 = rank(Cc2);

% det(lambda * Cc1 + mu * Cc2) = 0.
CC = {};
kk = 1;
if rankCc1 >= 3
	% mu = 1 -> det(lambda * Cc1 + Cc2) = 0;

	a = 8*A1*C1*F1-2*B1^2*F1-2*A1*E1^2+2*B1*D1*E1-2*C1*D1^2;
	b = 8*A1*C1*F2-2*B1^2*F2+8*A1*C2*F1+8*A2*C1*F1-4*B1*B2*F1-4*A1*E1*E2 ...
		+2*B1*D1*E2-2*A2*E1^2+2*B1*D2*E1+2*B2*D1*E1-4*C1*D1*D2-2*C2*D1^2;
	c = 8*A1*C2*F2+8*A2*C1*F2-4*B1*B2*F2+8*A2*C2*F1-2*B2^2*F1-2*A1*E2^2 ...
		-4*A2*E1*E2+2*B1*D2*E2+2*B2*D1*E2+2*B2*D2*E1-2*C1*D2^2-4*C2*D1*D2;
	d = 8*A2*C2*F2-2*B2^2*F2-2*A2*E2^2+2*B2*D2*E2-2*C2*D2^2;

	lambda = roots([a b c d]);

	%CC = cell(size(lambda));
	for ii = 1:length(lambda)
	%for ii = 1:2
		CC{kk} = lambda(ii) * Cc1 + Cc2;  % rank 2.
		kk = kk + 1;
	end;
end;
if rankCc2 >= 3
	% lambda = 1 -> det(Cc1 + mu * Cc2) = 0;

	a = 8*A2*C2*F2-2*B2^2*F2-2*A2*E2^2+2*B2*D2*E2-2*C2*D2^2;
	b = 8*A1*C2*F2+8*A2*C1*F2-4*B1*B2*F2+8*A2*C2*F1-2*B2^2*F1-2*A1*E2^2 ...
		-4*A2*E1*E2+2*B1*D2*E2+2*B2*D1*E2+2*B2*D2*E1-2*C1*D2^2-4*C2*D1*D2;
	c = 8*A1*C1*F2-2*B1^2*F2+8*A1*C2*F1+8*A2*C1*F1-4*B1*B2*F1-4*A1*E1*E2 ...
		+2*B1*D1*E2-2*A2*E1^2+2*B1*D2*E1+2*B2*D1*E1-4*C1*D1*D2-2*C2*D1^2;
	d = 8*A1*C1*F1-2*B1^2*F1-2*A1*E1^2+2*B1*D1*E1-2*C1*D1^2;

	mu = roots([a b c d]);

	%CC = cell(size(mu));
	for ii = 1:length(mu)
	%for ii = 1:2
		CC{kk} = Cc1 + mu(ii) * Cc2;  % rank 2.
		kk = kk + 1;
	end;
end;

LL = cell(1, length(CC));
for ii = 1:length(CC)
	%adjCC = det(CC{ii}) * inv(CC{ii});  % rank 1.
	adjCC = adjoint(CC{ii});  % rank 1.

	p = [];
	for jj = 1:size(adjCC, 2)
		if norm(adjCC(:,jj)) > tol
			p = adjCC(:,jj);
			break;
		end;
	end;
	if isempty(p)
		error('[Error] A valid point not found.');
	end;

	P = [
		0 p(3) -p(2)
		-p(3) 0 p(1)
		p(2) -p(1) 0
	];

	alpha = sqrt((CC{ii}(1,2)^2 - CC{ii}(1,1) * CC{ii}(2,2)) / p(3)^2);
	%alpha = -sqrt((CC{ii}(1,2)^2 - CC{ii}(1,1) * CC{ii}(2,2)) / p(3)^2);

	LL{ii} = zeros(size(P,1),4);
	for mm = 1:2
		if 1 == mm
			CP = CC{ii} + alpha * P;
		else
			CP = CC{ii} - alpha * P;
		end;

	    for jj = 1:size(CP, 1)
			if norm(CP(jj,:)) > tol
				LL{ii}(:,2*(mm-1)+1) = CP(jj,:)';
				break;
			end;
		end;
		for jj = 1:size(CP, 2)
			if norm(CP(:,jj)) > tol
				LL{ii}(:,2*(mm-1)+2) = CP(:,jj);
				break;
			end;
		end;
		if isempty(LL{ii}(:,1)) | isempty(LL{ii}(:,2)) | isempty(LL{ii}(:,3)) | isempty(LL{ii}(:,4))
			error('[Error] Any valid line not found.');
		end;
	end;
end;

pts = [
	cross(LL{1}(:,1), LL{2}(:,1)) cross(LL{1}(:,1), LL{2}(:,2)) cross(LL{1}(:,2), LL{2}(:,1)) cross(LL{1}(:,2), LL{2}(:,2)) ...
	cross(LL{1}(:,1), LL{3}(:,1)) cross(LL{1}(:,1), LL{3}(:,2)) cross(LL{1}(:,2), LL{3}(:,1)) cross(LL{1}(:,2), LL{3}(:,2)) ...
	cross(LL{2}(:,1), LL{3}(:,1)) cross(LL{2}(:,1), LL{3}(:,2)) cross(LL{2}(:,2), LL{3}(:,1)) cross(LL{2}(:,2), LL{3}(:,2)) ...
	cross(LL{1}(:,3), LL{2}(:,3)) cross(LL{1}(:,3), LL{2}(:,4)) cross(LL{1}(:,4), LL{2}(:,3)) cross(LL{1}(:,4), LL{2}(:,4)) ...
	cross(LL{1}(:,3), LL{3}(:,3)) cross(LL{1}(:,3), LL{3}(:,4)) cross(LL{1}(:,4), LL{3}(:,3)) cross(LL{1}(:,4), LL{3}(:,4)) ...
	cross(LL{2}(:,3), LL{3}(:,3)) cross(LL{2}(:,3), LL{3}(:,4)) cross(LL{2}(:,4), LL{3}(:,3)) cross(LL{2}(:,4), LL{3}(:,4)) ...
	cross(LL{4}(:,1), LL{5}(:,1)) cross(LL{4}(:,1), LL{5}(:,2)) cross(LL{4}(:,2), LL{5}(:,1)) cross(LL{4}(:,2), LL{5}(:,2)) ...
	cross(LL{4}(:,1), LL{6}(:,1)) cross(LL{4}(:,1), LL{6}(:,2)) cross(LL{4}(:,2), LL{6}(:,1)) cross(LL{4}(:,2), LL{6}(:,2)) ...
	cross(LL{5}(:,1), LL{6}(:,1)) cross(LL{5}(:,1), LL{6}(:,2)) cross(LL{5}(:,2), LL{6}(:,1)) cross(LL{5}(:,2), LL{6}(:,2)) ...
	cross(LL{4}(:,3), LL{5}(:,3)) cross(LL{4}(:,3), LL{5}(:,4)) cross(LL{4}(:,4), LL{5}(:,3)) cross(LL{4}(:,4), LL{5}(:,4)) ...
	cross(LL{4}(:,3), LL{6}(:,3)) cross(LL{4}(:,3), LL{6}(:,4)) cross(LL{4}(:,4), LL{6}(:,3)) cross(LL{4}(:,4), LL{6}(:,4)) ...
	cross(LL{5}(:,3), LL{6}(:,3)) cross(LL{5}(:,3), LL{6}(:,4)) cross(LL{5}(:,4), LL{6}(:,3)) cross(LL{5}(:,4), LL{6}(:,4))
];

sols = [];
kk = 1;
for ii = 1:size(pts, 2)
	pts(:,ii) = pts(:,ii) / pts(3,ii);
	if abs(transpose(pts(:,ii)) * Cc1 * pts(:,ii)) < tol & abs(transpose(pts(:,ii)) * Cc2 * pts(:,ii)) < tol
		sols(:,kk) = pts(:,ii);
		kk = kk + 1;
	end;
end;

if ~isempty(sols)
	sols = find_unique(sols, tol);
	sols = sols(1:2,:);
end;

if 4 ~= size(sols,2)
	error('[Error] Invalid number of solutions.');
end;

return;

%-----------------------------------------------------------

function adjA = adjoint(A)

adjA = [
	A(2,2)*A(3,3)-A(2,3)^2 A(1,3)*A(2,3)-A(1,2)*A(3,3) A(1,2)*A(2,3)-A(1,3)*A(2,2)
	A(1,3)*A(2,3)-A(1,2)*A(3,3) A(1,1)*A(3,3)-A(1,3)^2 A(1,2)*A(1,3)-A(1,1)*A(2,3)
	A(1,2)*A(2,3)-A(1,3)*A(2,2) A(1,2)*A(1,3)-A(1,1)*A(2,3) A(1,1)*A(2,2)-A(1,2)^2
];

return;
