function [x,y,Z] = plot_THoG(HoG_sequence, plot_type)

% plot a surface of temporal HoG (THoG)

[ numFeatures numFrames ] = size(HoG_sequence);

x = 1:numFrames;
y = 0:(numFeatures - 1);
Z = HoG_sequence;

switch plot_type
	case 1
		mesh(x, y, Z);
	case 2
		meshc(x, y, Z);
	case 3
		surf(x, y, Z);
	case 4
		surfc(x, y, Z);
	case 5
		contour(x, y, Z);
	otherwise
    	error('plot type is not defined');
end;
