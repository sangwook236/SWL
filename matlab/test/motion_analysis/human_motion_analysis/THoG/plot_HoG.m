function [ h ] = plot_HoG(HoG, HoG_bin_width, HoG_scale_factor, plot_type)

% plot a mixture of histograms of oriented gradients (HoG)
%
%  random angle data [0 2*pi) [rad].

angleData = HoG_to_angle(HoG, HoG_bin_width, HoG_scale_factor);
num_samples = length(angleData);

if 1 == plot_type
    %----------------------------------------------------------
    % approach #1: use direction angles, [rad].

	[binheight, bincenter] = hist(angleData, 360);
	binwidth = bincenter(2) - bincenter(1);
	x_rng = [ 0 2*pi ];
	area = num_samples * binwidth;

	h = bar(bincenter, binheight / area, 'hist');
	set(h, 'facecolor', [0.8 0.8 1]);
    axis_rng = axis;
    axis([ x_rng(1) x_rng(2) axis_rng(3) axis_rng(4) ]);

	hold on;
	xi = linspace(x_rng(1), x_rng(2), 361);
	f = ksdensity(angleData, xi, 'kernel', 'normal', 'function', 'pdf');
	plot(xi, f, 'b-', 'linewidth', 2);
	hold off;
elseif 2 == plot_type
    %----------------------------------------------------------
    % approach #2: use direction angles, [rad].

    x_rng = [ 0 2*pi ];
	[binheight, bincenter] = hist(angleData, 360);
    subplot(2,1,1), h = bar(bincenter, binheight, 'hist');
    axis_rng = axis;
    axis([ x_rng(1) x_rng(2) axis_rng(3) axis_rng(4) ]);

	xi = linspace(x_rng(1), x_rng(2), 361);
	f = ksdensity(angleData, xi, 'kernel', 'normal', 'function', 'pdf');
	subplot(2,1,2), plot(xi, f, 'b-', 'linewidth', 2);
    axis_rng = axis;
    axis([ x_rng(1) x_rng(2) axis_rng(3) axis_rng(4) ]);
else
    error('plot type is not defined');
end;
