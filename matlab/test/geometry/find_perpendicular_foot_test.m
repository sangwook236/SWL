%addpath('D:/work/SWL_github/matlab/src/geometry');

%-----------------------------------------------------------

line = [1 0 0]';  % x = 0.
pt = [1 3 1]';  % (1, 3).
int_pt = find_perpendicular_foot(line, pt)

line = [0 1 2]';  % y + 2 = 0.
pt = [1 3 1]';  % (1, 3).
int_pt = find_perpendicular_foot(line, pt)

line = [1 -1 0]';  % x - y = 0.
pt = [0 1 1]';  % (0, 1).
int_pt = find_perpendicular_foot(line, pt)
