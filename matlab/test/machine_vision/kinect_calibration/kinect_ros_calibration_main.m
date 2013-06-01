%----------------------------------------------------------
% [ref]
%	[1] http://www.ros.org/wiki/kinect_calibration/technical
%	[2] http://www.ros.org/wiki/kinect_node/Calibration

%----------------------------------------------------------

% from [1]
b = 0.075;
doff = 1090;
fir = 580;

x = linspace(0, 2047, 1000);
y1 = (-x + doff) / (8 * b * fir);
y2 = -0.00307110156374373*x+3.33094951605675;

plot(x, y1, 'r-', x, y2, 'b-');
