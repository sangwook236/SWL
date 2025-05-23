%
% using Paul's notation
%	Fu's book pp. 37
%

%addpath('../../src/robot_kinematics');

notation = 'paul';

%------------------------------------------------------------------------------
a2 = 431.8;
a3 = 20.32;
d2 = 149.09;
d4 = 433.07;
d6 = 56.25;

%------------------------------------------------------------------------------
T0 = eye(4);
T1 = [
	0	0	-1	0
	1	0	0	0
	0	-1	0	0
	0	0	0	1
];
T2 = [
	0	0	-1	-d2
	1	0	0	a2
	0	-1	0	0
	0	0	0	1
];
T3 = [
	0	-1	0	-d2
	0	0	1	a2
	-1	0	0	a3
	0	0	0	1
];
T4 = [
	0	0	-1	-d2
	0	-1	0	a2 + d4
	-1	0	0	a3
	0	0	0	1
];
T5 = [
	0	-1	0	-d2
	0	0	1	a2 + d4
	-1	0	0	a3
	0	0	0	1
];
T6 = [
	0	-1	0	-d2
	0	0	1	a2 + d4 + d6
	-1	0	0	a3
	0	0	0	1
];

T_init = cell(1, 7);
T_init{1} = T0;
T_init{2} = T1;
T_init{3} = T2;
T_init{4} = T3;
T_init{5} = T4;
T_init{6} = T5;
T_init{7} = T6;

%------------------------------------------------------------------------------

% [ a alpha d theta ]
dh_param = calc_dh_param(T_init, notation);

%------------------------------------------------------------------------------
joint_type = [ 1 1 1 1 1 1 ];
axis_length = 100;

figure;
%qq = [-135:1:135] * pi / 180;
qq = [0];
for ii = 1:length(qq)
	clf;

	q = [ qq(ii) 0 0 0 0 0 ];
	T = calc_robot_pose(dh_param, notation, joint_type, q);

	draw_robot_frame(T, axis_length);
end;
