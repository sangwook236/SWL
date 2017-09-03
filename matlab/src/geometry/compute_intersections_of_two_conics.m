function sols = compute_intersections_of_two_conics(ABCDEF1, ABCDEF2)
% A1*x^2 + B1*x*y + C1*y^2 + D1*x + E1*y + F1 = 0.
% A2*x^2 + B2*x*y + C2*y^2 + D2*x + E2*y + F2 = 0.
% REF [site] >> https://elliotnoma.wordpress.com/2013/04/10/a-closed-form-solution-for-the-intersections-of-two-ellipses/

sols = compute_intersections_of_two_conics1(ABCDEF1, ABCDEF2);
%sols = compute_intersections_of_two_conics2(ABCDEF1, ABCDEF2);
%sols = compute_intersections_of_two_conics3(ABCDEF1, ABCDEF2);
return;

function sols = compute_intersections_of_two_conics1(ABCDEF1, ABCDEF2)
% A1*x^2 + B1*x*y + C1*y^2 + D1*x + E1*y + F1 = 0.
% A2*x^2 + B2*x*y + C2*y^2 + D2*x + E2*y + F2 = 0.

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

if abs(aa) > eps
	determinant = sqrt(bb.^2 - 4 * aa * cc);
	yy = [(-bb + determinant) (-bb - determinant)] / (2 * aa);
	func1 = abs(A2 * x.^2 + B2 * x.*yy(:,1) + C2 * yy(:,1).^2 + D2 * x + E2 * yy(:,1) + F2);
	func2 = abs(A2 * x.^2 + B2 * x.*yy(:,2) + C2 * yy(:,2).^2 + D2 * x + E2 * yy(:,2) + F2);
	[X, I] = min([func1 func2], [], 2);
	y = zeros(size(x));
	for ii = 1:length(I)
		y(ii) = yy(ii,I(ii));
	end;
elseif all(abs(bb) > eps)
	y = -cc ./ bb;
else
	error('[Error] Improper quadratic equation.')
end;

sols = [x, y];
return;

function sols = compute_intersections_of_two_conics2(ABCDEF1, ABCDEF2)
% A1*x^2 + B1*x*y + C1*y^2 + D1*x + E1*y + F1 = 0.
% A2*x^2 + B2*x*y + C2*y^2 + D2*x + E2*y + F2 = 0.
% REF [site] >> https://elliotnoma.wordpress.com/2013/04/10/a-closed-form-solution-for-the-intersections-of-two-ellipses/

a = ABCDEF1(1);
b = ABCDEF1(2);
c = ABCDEF1(3);
d = ABCDEF1(4);
e = ABCDEF1(5);
f = ABCDEF1(6);

A1 = ABCDEF2(1);
B1 = ABCDEF2(2);
C1 = ABCDEF2(3);
D1 = ABCDEF2(4);
E1 = ABCDEF2(5);
F1 = ABCDEF2(6);

% z0 + z1 * y + z2 * y^2 + z3 * y^3 + z4 * y^4 = 0.
z0 = f*a*D1^2+a^2*F1^2-d*a*D1*F1+A1^2*f^2-2*a*F1*A1*f-d*D1*A1*f+A1*d^2*F1;
z1 = E1*d^2*A1-F1*D1*a*b-2*a*F1*A1*e-f*A1*B1*d+2*D1*B1*a*f+2*E1*F1*a^2+D1^2*a*e...
	-E1*D1*a*d-2*a*E1*A1*f-f*A1*D1*b+2*f*e*A1^2-F1*B1*a*d-e*A1*D1*d+2*F1*b*A1*d;
z2 = E1^2*a^2+2*C1*F1*a^2-e*A1*D1*b+F1*A1*b^2-e*A1*B1*d-F1*B1*a*b-2*a*E1*A1*e...
	+2*D1*B1*a*e-C1*D1*a*d-2*a*C1*A1*f+B1^2*a*f+2*E1*b*A1*d+e^2*A1^2-c*A1*D1*d...
	-E1*B1*a*d+2*f*c*A1^2-f*A1*B1*b+C1*d^2*A1+D1^2*a*c-E1*D1*a*b-2*a*F1*A1*c;
z3 = -2*a*A1*c*E1+E1*A1*b^2+2*C1*b*A1*d-c*A1*B1*d+B1^2*a*e-E1*B1*a*b-2*a*C1*A1*e...
	-e*A1*B1*b-C1*B1*a*d+2*E1*C1*a^2+2*e*c*A1^2-c*A1*D1*b+2*D1*B1*a*c-C1*D1*a*b;
z4 = a^2*C1^2-2*a*C1*A1*c+A1^2*c^2-b*a*B1*C1-b*B1*A1*c+b^2*A1*C1+c*a*B1^2;

y = roots([z4 z3 z2 z1 z0]);

denom = a*B1*y+a*D1-A1*b*y-A1*d;
if abs(denom) > eps
	x = -(a*F1+a*C1*y.^2-A1*c*y.^2+a*E1*y-A1*e*y-A1*f) ./ denom;
else
	bb = b * y + d;
	cc = c * y.^2 + e * y + f;
	x1 = (-bb + sqrt(bb.^2 - 4 * a * cc)) / (2 * a);
	x1
	x = (-bb - sqrt(bb.^2 - 4 * a * cc)) / (2 * a);
end;

sols = [x, y];
return;

function sols = compute_intersections_of_two_conics3(ABCDEF1, ABCDEF2)
% A1*x^2 + B1*x*y + C1*y^2 + D1*x + E1*y + F1 = 0.
% A2*x^2 + B2*x*y + C2*y^2 + D2*x + E2*y + F2 = 0.

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

A = [
	2*A1 B1 D1
	B1 2*C1 E1
	D1 E1 2*F1
];
B = [
	2*A2 B2 D2
	B2 2*C2 E2
	D2 E2 2*F2
];

rankA = rank(A);
rankB = rank(B);

% det(lambda * A + mu * B) = 0;
%if rankA >= 3
if rankA >= 5
	% mu = 1 -> det(lambda * A + B) = 0;

	a = 8*A1*C1*F1-2*B1^2*F1-2*A1*E1^2+2*B1*D1*E1-2*C1*D1^2;
	b = 8*A1*C1*F2-2*B1^2*F2+8*A1*C2*F1+8*A2*C1*F1-4*B1*B2*F1-4*A1*E1*E2 ...
		+2*B1*D1*E2-2*A2*E1^2+2*B1*D2*E1+2*B2*D1*E1-4*C1*D1*D2-2*C2*D1^2;
	c = 8*A1*C2*F2+8*A2*C1*F2-4*B1*B2*F2+8*A2*C2*F1-2*B2^2*F1-2*A1*E2^2 ...
		-4*A2*E1*E2+2*B1*D2*E2+2*B2*D1*E2+2*B2*D2*E1-2*C1*D2^2-4*C2*D1*D2;
	d = 8*A2*C2*F2-2*B2^2*F2-2*A2*E2^2+2*B2*D2*E2-2*C2*D2^2;

	lambda = roots([a b c d]);

	CC = cell(size(lambda));
	for ii = 1:length(lambda)
	%for ii = 1:2
		CC{ii} = lambda(ii) * A + B;  % rank 2.
	end;
elseif rankB >= 3
	% lambda = 1 -> det(A + mu * B) = 0;

	a = 8*A2*C2*F2-2*B2^2*F2-2*A2*E2^2+2*B2*D2*E2-2*C2*D2^2;
	b = 8*A1*C2*F2+8*A2*C1*F2-4*B1*B2*F2+8*A2*C2*F1-2*B2^2*F1-2*A1*E2^2 ...
		-4*A2*E1*E2+2*B1*D2*E2+2*B2*D1*E2+2*B2*D2*E1-2*C1*D2^2-4*C2*D1*D2;
	c = 8*A1*C1*F2-2*B1^2*F2+8*A1*C2*F1+8*A2*C1*F1-4*B1*B2*F1-4*A1*E1*E2 ...
		+2*B1*D1*E2-2*A2*E1^2+2*B1*D2*E1+2*B2*D1*E1-4*C1*D1*D2-2*C2*D1^2;
	d = 8*A1*C1*F1-2*B1^2*F1-2*A1*E1^2+2*B1*D1*E1-2*C1*D1^2;

	mu = roots([a b c d]);

	CC = cell(size(mu));
	for ii = 1:length(mu)
	%for ii = 1:2
		CC{ii} = A + mu(ii) * B;  % rank 2.
	end;
end;

l1 = cell(1, length(CC));
l2 = cell(1, length(CC));
for ii = 1:length(CC)
	adjCC = det(CC{ii}) * inv(CC{ii});  % rank 1.

	p = [];
	for jj = 1:size(adjCC, 2)
		if norm(adjCC(:,jj)) > eps
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

	CP = CC{ii} + alpha * P;

    for jj = 1:size(CP, 1)
		if norm(CP(jj,:)) > eps
			l1{ii} = CP(jj,:)';
			break;
		end;
	end;
	for jj = 1:size(CP, 2)
		if norm(CP(:,jj)) > eps
			l2{ii} = CP(:,jj);
			break;
		end;
	end;
	if isempty(l1{ii}) | isempty(l2{ii})
		error('[Error] Any valid line not found.');
	end;
end;

l1{1}
l1{2}
l1{3}
l2{1}
l2{2}
l2{3}

pt1 = cross(l1{1}, l2{1})
pt2 = cross(l1{1}, l2{2});
%pt3 = cross(l1{1}, l2{3});
pt4 = cross(l1{2}, l2{1});
pt5 = cross(l1{2}, l2{2});
%pt6 = cross(l1{2}, l2{3});
%pt7 = cross(l1{3}, l2{1});
%pt8 = cross(l1{3}, l2{2});
%pt9 = cross(l1{3}, l2{3});

sols = [];
return;
