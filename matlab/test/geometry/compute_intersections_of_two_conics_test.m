%addpath('D:/work/SWL_github/matlab/src');
%addpath('D:/work/SWL_github/matlab/src/geometry');

%-----------------------------------------------------------
% Two ellipses w/ four intersection points.

[a, b, theta, tx, ty] = deal(3, 1, pi / 6, 1, 0);
ABCDEF1 = ellipse2conic(a, b, theta, tx, ty);
[a, b, theta, tx, ty] = deal(4, 2, -pi / 3, 2, -1);
ABCDEF2 = ellipse2conic(a, b, theta, tx, ty);

%figure;
%hold on;
%draw_conic(ABCDEF1, 'r-');
%draw_conic(ABCDEF2, 'g-');
%axis equal;
%hold off;

% (0.02102206343458029,-1.53788290165648).
% (-0.5611929268108435,0.1428565595821076).
% (2.432306227373992,1.662174402330793).
% (3.333982353715173,0.6210199363163925).
sols1_1 = compute_intersections_of_two_conics(ABCDEF1, ABCDEF2)

%-----------------------------------------------------------
% Two ellipses w/ two intersection points.

[a, b, theta, tx, ty] = deal(2, 1, pi / 6, 1, 0);
ABCDEF1 = ellipse2conic(a, b, theta, tx, ty);
[a, b, theta, tx, ty] = deal(4, 2, -pi / 3, 2, -1);
ABCDEF2 = ellipse2conic(a, b, theta, tx, ty);

%figure;
%hold on;
%draw_conic(ABCDEF1, 'r-');
%draw_conic(ABCDEF2, 'g-');
%axis equal;
%hold off;

% (-0.08221582906723197,-1.319832191661396).
% (-0.5268728164987108,-0.0204712434295121).
% (2.889715531827274+0.3483797175702664*i,1.218313841834484-0.3992365066786316*i).
% (2.889715531827274-0.3483797175702664*i,1.218313841834482+0.3992365066786326*i).
sols1_2 = compute_intersections_of_two_conics(ABCDEF1, ABCDEF2)

%-----------------------------------------------------------
% Two ellipses w/ no intersection point.

[a, b, theta, tx, ty] = deal(5, 3, pi / 6, 1, 0);
ABCDEF1 = ellipse2conic(a, b, theta, tx, ty);
[a, b, theta, tx, ty] = deal(2, 1, -pi / 3, 2, 1);
ABCDEF2 = ellipse2conic(a, b, theta, tx, ty);

%figure;
%hold on;
%draw_conic(ABCDEF1, 'r-');
%draw_conic(ABCDEF2, 'g-');
%axis equal;
%hold off;

% (0.8302415405925176+0.7561382649674434*i,3.259364432456434+0.2692144539208489*i).
% (0.8302415405925176-0.7561382649674434*i,3.259364432456434-0.2692144539208489*i).
% (3.805986690314117+1.040846913238235*i,-1.820938399351997+0.8859023514080547*i).
% (3.805986690314117-1.040846913238235*i,-1.820938399351997-0.8859023514080546*i).
sols1_3 = compute_intersections_of_two_conics(ABCDEF1, ABCDEF2)

%-----
[a, b, theta, tx, ty] = deal(4, 2, pi / 6, 1, 0);
ABCDEF1 = ellipse2conic(a, b, theta, tx, ty);
[a, b, theta, tx, ty] = deal(2, 1, -pi / 3, 6, -2);
ABCDEF2 = ellipse2conic(a, b, theta, tx, ty);

%figure;
%hold on;
%draw_conic(ABCDEF1, 'r-');
%draw_conic(ABCDEF2, 'g-');
%axis equal;
%hold off;

% (4.335038589351573+0.5948677470794136*i,0.1996846668043205+0.9006785157418673*i).
% (4.335038589351573-0.5948677470794136*i,0.1996846668043206-0.9006785157418673*i).
% (3.535303828737033+3.338985594195218*i,3.841125954640962+0.2009114009096226*i).
% (3.535303828737033-3.338985594195218*i,3.841125954640961-0.20091140090962*i).
sols1_4 = compute_intersections_of_two_conics(ABCDEF1, ABCDEF2)

%-----------------------------------------------------------
% Two axis-aligned ellipses w/ four intersection points.

ABCDEF1 = [1 0 4 0 8 -12];  % x^2 / 16 + (y - 1)^2 / 4 = 1.
ABCDEF2 = [6 0 1 0 4 -8];  % x^2 / 2 + (y - 2)^2 = 1.

%figure;
%hold on;
%draw_conic(ABCDEF1, 'r-');
%draw_conic(ABCDEF2, 'g-');
%axis equal;
%hold off;

% (0.730365, 0.97), (-0.730365, 0.97), (1.36788, -2.88), (-1.36788, -2.88).
sols2_1 = compute_intersections_of_two_conics(ABCDEF1, ABCDEF2)

%-----------------------------------------------------------
% Two axis-aligned ellipses w/ no intersection point.

ABCDEF1 = [4 0 1 0 -2 -15];  % x^2 / 4 + (y - 1)^2 / 16 = 1.
ABCDEF2 = [1 0 2 0 -8 6];  % x^2 / 2 + (y - 2)^2 = 1.

%figure;
%hold on;
%draw_conic(ABCDEF1, 'r-');
%draw_conic(ABCDEF2, 'g-');
%axis equal;
%hold off;

% (1.984608351965436+0.1424884688393677*i,2.142857142857157-0.9897433186108037*i).
% (1.984608351965436-0.1424884688393677*i,2.142857142857147+0.9897433186107877*i).
% (-1.984608351965436+0.1424884688393676*i,2.142857142857157+0.9897433186108034*i).
% (-1.984608351965436-0.1424884688393676*i,2.142857142857157-0.9897433186108034*i).
sols2_2 = compute_intersections_of_two_conics(ABCDEF1, ABCDEF2)

%-----------------------------------------------------------
% An ellipse & a circle w/ four intersection points.

[a, b, theta, tx, ty] = deal(3, 1, pi / 4, 0, 0);
ABCDEF1 = ellipse2conic(a, b, theta, tx, ty);
ABCDEF2 = [1 0 1 0 0 -4];  % x^2 + y^2 = 4.

%figure;
%hold on;
%draw_conic(ABCDEF1, 'r-');
%draw_conic(ABCDEF2, 'g-');
%axis equal;
%hold off;

% (-0.7400, -1.8581), (0.7400, 1.8581), (-1.8581, -0.7400), (1.8581, 0.7400).
sols3_1 = compute_intersections_of_two_conics(ABCDEF1, ABCDEF2)

%-----------------------------------------------------------
% An ellipse & a circle w/ three intersection points (a point of contact).

[a, b, theta, tx, ty] = deal(3, 1, pi / 4, 0, 0);
ABCDEF1 = ellipse2conic(a, b, theta, tx, ty);
ABCDEF2 = [1 0 1 sqrt(2) -sqrt(2) -3];  % (x + sqrt(2)/2)^2 + (y - sqrt(2)/2)^2 = 4.

%figure;
%hold on;
%draw_conic(ABCDEF1, 'r-');
%draw_conic(ABCDEF2, 'g-');
%axis equal;
%hold off;

sols3_2 = compute_intersections_of_two_conics(ABCDEF1, ABCDEF2)

%-----------------------------------------------------------
% An ellipse & a circle w/ two intersection points.

% Two regular intersection points.
[a, b, theta, tx, ty] = deal(3, 1, pi / 4, 0, 0);
ABCDEF1 = ellipse2conic(a, b, theta, tx, ty);
ABCDEF2 = [1 0 1 2 -2 -3];  % (x + 1)^2 + (y - 1)^2 = 4.

%figure;
%hold on;
%draw_conic(ABCDEF1, 'r-');
%draw_conic(ABCDEF2, 'g-');
%axis equal;
%hold off;

sols3_3 = compute_intersections_of_two_conics(ABCDEF1, ABCDEF2)

%-----
% One point of contact & one regular intersection point.
ABCDEF1 = [1 0 1 0 0 -1];  % x^2 + y^2 - 1 = 0.
ABCDEF2 = [5 6 5 0 6 -5];  % 5 * x^2 + 6 * x * y + 5 * y^2 + 6 * y - 5 = 0.

%figure;
%hold on;
%draw_conic(ABCDEF1, 'r-');
%draw_conic(ABCDEF2, 'g-');
%axis equal;
%hold off;

% (-1, 0), (1, 0).
%	(-1, 0) => an intersection of multiplicity 3 (a point of tangency).
sols3_4 = compute_intersections_of_two_conics(ABCDEF1, ABCDEF2)

%-----
% Two points of contact.
ABCDEF1 = [1 0 1 0 0 -1];  % x^2 + y^2 - 1 = 0.
ABCDEF2 = [1 0 4 0 0 -1];  % x^2 + 4 * y^2 - 1 = 0.

%figure;
%hold on;
%draw_conic(ABCDEF1, 'r-');
%draw_conic(ABCDEF2, 'g-');
%axis equal;
%hold off;

% (-1, 0), (1, 0) => two intersections of multiplicity 2 (two points of tangency).
sols3_5 = compute_intersections_of_two_conics(ABCDEF1, ABCDEF2)

%-----------------------------------------------------------
% An ellipse & a circle w/ one intersection point.

% One (outer) point of contact.
[a, b, theta, tx, ty] = deal(3, 1, pi / 4, 0, 0);
ABCDEF1 = ellipse2conic(a, b, theta, tx, ty);
ABCDEF2 = [1 0 1 3*sqrt(2) -3*sqrt(2) 5];  % (x + 3*sqrt(2)/2)^2 + (y - 3*sqrt(2)/2)^2 = 4.

%figure;
%hold on;
%draw_conic(ABCDEF1, 'r-');
%draw_conic(ABCDEF2, 'g-');
%axis equal;
%hold off;

% One (inner) point of contact.
sols3_6 = compute_intersections_of_two_conics(ABCDEF1, ABCDEF2)

%-----
% One (inner) point of contact.
[a, b, theta, tx, ty] = deal(3, 1, pi / 4, 0, 0);
ABCDEF1 = ellipse2conic(a, b, theta, tx, ty);
ABCDEF2 = [1 0 1 sqrt(2)/2 -sqrt(2)/2 0];  % (x + sqrt(2)/4)^2 + (y - sqrt(2)/4)^2 = 1/4.

%figure;
%hold on;
%draw_conic(ABCDEF1, 'r-');
%draw_conic(ABCDEF2, 'g-');
%axis equal;
%hold off;

% One (inner) point of contact.
sols3_7 = compute_intersections_of_two_conics(ABCDEF1, ABCDEF2)

%-----
% One point of contact.
ABCDEF1 = [1 0 1 0 0 -1];  % x^2 + y^2 - 1 = 0.
ABCDEF2 = [4 0 1 6 0 2];  % 4 * x^2 + y^2 + 6 * x + 2 = 0.

%figure;
%hold on;
%draw_conic(ABCDEF1, 'r-');
%draw_conic(ABCDEF2, 'g-');
%axis equal;
%hold off;

% (-1, 0) => an intersection of multiplicity 4 (a point of tangency).
sols3_8 = compute_intersections_of_two_conics(ABCDEF1, ABCDEF2)

%-----------------------------------------------------------
% An ellipse & a circle w/ no intersection point.

[a, b, theta, tx, ty] = deal(3, 1, pi / 4, 0, 0);
ABCDEF1 = ellipse2conic(a, b, theta, tx, ty);
ABCDEF2 = [1 0 1 -2 -2 7/4];  % (x - 1)^2 + (y - 1)^2 = 1/4.

%figure;
%hold on;
%draw_conic(ABCDEF1, 'r-');
%draw_conic(ABCDEF2, 'g-');
%axis equal;
%hold off;

sols3_9 = compute_intersections_of_two_conics(ABCDEF1, ABCDEF2)

%-----
[a, b, theta, tx, ty] = deal(3, 1, pi / 4, 0, 0);
ABCDEF1 = ellipse2conic(a, b, theta, tx, ty);
ABCDEF2 = [1 0 1 6 -4 12];  % (x + 3)^2 + (y - 2)^2 = 1.

%figure;
%hold on;
%draw_conic(ABCDEF1, 'r-');
%draw_conic(ABCDEF2, 'g-');
%axis equal;
%hold off;

sols3_10 = compute_intersections_of_two_conics(ABCDEF1, ABCDEF2)

%-----------------------------------------------------------
% Two circles w/ two intersection points.

ABCDEF1 = [1 0 1 0 0 -1];  % x^2 + y^2 = 1.
ABCDEF2 = [1 0 1 -2 -2 1];  % (x - 1)^2 + (y - 1)^2 = 1.

%figure;
%hold on;
%draw_conic(ABCDEF1, 'r-');
%draw_conic(ABCDEF2, 'g-');
%axis equal;
%hold off;

% (1, 0), (0, 1).
%	(1, i), (1, -i) => the circular points at infinity.
sols4_1 = compute_intersections_of_two_conics(ABCDEF1, ABCDEF2)

%-----------------------------------------------------------
% Two concentric circles w/ no intersection point.

ABCDEF1 = [1 0 1 -2 -2 1];  % (x - 1)^2 + (y - 1)^2 = 1.
ABCDEF2 = [1 0 1 -2 -2 -2];  % (x - 1)^2 + (y - 1)^2 = 4.

%figure;
%hold on;
%draw_conic(ABCDEF1, 'r-');
%draw_conic(ABCDEF2, 'g-');
%axis equal;
%hold off;

% (1, i), (1, -i) => the circular points at infinity with an intersection multiplicity of two.
sols14_2 = compute_intersections_of_two_conics(ABCDEF1, ABCDEF2)

%-----------------------------------------------------------
% Two (non-concentric) circles w/ no intersection point.

ABCDEF1 = [1 0 1 -2 -2 1];  % (x - 1)^2 + (y - 1)^2 = 1.
ABCDEF2 = [1 0 1 -2 0 -8];  % (x - 1)^2 + y^2 = 9.

%figure;
%hold on;
%draw_conic(ABCDEF1, 'r-');
%draw_conic(ABCDEF2, 'g-');
%axis equal;
%hold off;

% (1.0000-3.3541i,4.5000+0.0000i), (1.0000+3.3541i,4.5000+0.0000i).
%	(1, i), (1, -i) => the circular points at infinity.
sols14_3 = compute_intersections_of_two_conics(ABCDEF1, ABCDEF2)
