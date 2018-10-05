% [ref] http://www.mathworks.com/matlabcentral/fileexchange/9896-2d-histogram-calculation

%addpath('D:\working_copy\swl_https\matlab\src\statistics');

events = 1000000; 
x1 = sqrt(0.05) * randn(events, 1) - 0.5;
x2 = sqrt(0.05) * randn(events, 1) + 0.5; 
y1 = sqrt(0.05) * randn(events, 1) + 0.5;
y2 = sqrt(0.05) * randn(events, 1) - 0.5; 

x = [x1 ; x2];
y = [y1 ; y2]; 

% for linearly spaced edges
xedges = linspace(-1, 1, 100);
yedges = linspace(-1, 1, 100); 
histmat = hist2(x, y, xedges, yedges); 

figure;
pcolor(xedges, yedges, histmat');
colorbar;
axis square tight; 

% for nonlinearly spaced edges
xedges2 = logspace(0, log10(3), 100) - 2;
yedges2 = linspace(-1, 1, 100); 
histmat2 = hist2(x, y, xedges2, yedges2); 

figure;
pcolor(xedges2, yedges2, histmat2');
colorbar;
axis square tight;
