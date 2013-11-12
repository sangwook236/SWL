function [x,y,Z] = plot_MovM_map(Mu, Kappa, Alpha, plot_type)

% plot a surface of mixtures of von Mises distributions (MovM)

[ numClusters numFrames ] = size(Mu);
[ numClusters2 numFrames2 ] = size(Kappa);
[ numClusters3 numFrames3 ] = size(Alpha);

if numClusters2 ~= numClusters || numClusters3 ~= numClusters
	error('the numbers of clusters is mis-matched');
end;
if numFrames2 ~= numFrames || numFrames3 ~= numFrames
	error('the numbers of frames is mis-matched');
end;

num_points = 1000;

x = 1:numFrames;
y = linspace(0, 2*pi, num_points);
Z = zeros(num_points, numFrames);

for jj = 1:numFrames
	Z(:,jj) = movm_pdf(y, Mu(:,jj), Kappa(:,jj), Alpha(:,jj));
end;

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
