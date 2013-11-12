function [ h1 h2 ] = plot_HoG_and_MovM(HoG, mu, kappa, alpha, HoG_bin_width, HoG_scale_factor, plot_type)

% plot a mixture of von Mises distributions (MovM)
%
%  random angle data [0 2*pi) [rad].

%numClusters = length(mu);
%numClusters2 = size(kappa);
%numClusters3 = size(alpha);

%if numClusters2 ~= numClusters || numClusters3 ~= numClusters
%	error('the numbers of clusters is mis-matched');
%end;

angleData = HoG_to_angle(HoG, HoG_bin_width, HoG_scale_factor);
num_samples = length(angleData);

if 1 == plot_type
    %----------------------------------------------------------
    % approach #1: use direction angles, [rad].

	[binheight, bincenter] = hist(angleData, 360);
	binwidth = bincenter(2) - bincenter(1);
	x_rng = [ 0 2*pi ];
	area1 = quad(@(theta) movm_pdf(theta, mu, kappa, alpha), x_rng(1), x_rng(2));
	area2 = num_samples * binwidth;

	% expectation of a function, g = E_f[g]
	% E_f[g] = 1/N sum(i=1 to N, g(x_i)) where x_i ~ movm_pdf(x)
	%E =  mean(g(smpl5));

	hold on;
	h1 = bar(bincenter, binheight / area2, 'hist');
	set(h1, 'facecolor', [0.8 0.8 1]);
	axis_rng = axis;

	xi = linspace(x_rng(1), x_rng(2), 361);
	f = ksdensity(angleData, xi, 'kernel', 'normal', 'function', 'pdf');
	plot(xi, f, 'b-', 'linewidth', 2);

	h2 = ezplot(@(theta) movm_pdf(theta, mu, kappa, alpha) / area1, x_rng);
	set(h2, 'color', [1 0 0], 'linewidth', 2);
	axis([x_rng(1) x_rng(2) axis_rng(3) axis_rng(4)]);
	hold off;
elseif 2 == plot_type
    %----------------------------------------------------------
    % approach #2: use direction angles, [rad].

	[binheight, bincenter] = hist(angleData, 360);
	binwidth = bincenter(2) - bincenter(1);
	x_rng = [ 0 2*pi ];
	area = quad(@(theta) movm_pdf(theta, mu, kappa, alpha), x_rng(1), x_rng(2));

	xi = linspace(x_rng(1), x_rng(2), 361);
	f = ksdensity(angleData, xi, 'kernel', 'normal', 'function', 'pdf');
	plot(xi, f, 'b-', 'linewidth', 2);

	hold on;
	h1 = bar(bincenter, binheight, 'hist');
	set(h1, 'facecolor', [0.8 0.8 1]);
	axis_rng = axis;

	h2 = ezplot(@(theta) (num_samples * binwidth / area) * movm_pdf(theta, mu, kappa, alpha), x_rng);
	set(h2, 'color', [1 0 0], 'linewidth', 2);
	axis([x_rng(1) x_rng(2) axis_rng(3) axis_rng(4)]);
	hold off;
elseif 3 == plot_type
    %----------------------------------------------------------
    % approach #3: use direction angles, [rad].

    x_rng = [ 0 2*pi ];
    subplot(3,1,1), h1 = hist(angleData, 360);
    axis_rng = axis;
    axis([ x_rng(1) x_rng(2) axis_rng(3) axis_rng(4) ]);

	xi = linspace(x_rng(1), x_rng(2), 361);
	f = ksdensity(angleData, xi, 'kernel', 'normal', 'function', 'pdf');
	subplot(3,1,2), plot(xi, f, 'b-', 'linewidth', 2);

    subplot(3,1,3), h2 = ezplot(@(theta) movm_pdf(theta, mu, kappa, alpha), x_rng)
else
    error('plot type is not defined');
end;
