% Line segment.
x_range = [-100 ; 100];
y_range = [-100 ; 100];
x1 = x_range(1) + (x_range(2) - x_range(1)) * rand();
x2 = x_range(1) + (x_range(2) - x_range(1)) * rand();
y1 = y_range(1) + (y_range(2) - y_range(1)) * rand();
y2 = y_range(1) + (y_range(2) - y_range(1)) * rand();
ls = [ x1 y1 x2 y2 ];
%ls = [ 0 0 100 100 ];

ref_len = 5;
subsegment = generate_evenly_divided_subsegment(ls, ref_len);
%subsegment = generate_subsegment_randomly(ls, ref_len);
%subsegment = generate_centered_subsegment(ls, ref_len);  % Not good.

figure;
axis equal;
line([ls(1) ls(3)], [ls(2) ls(4)], 'Color', 'red');
for ii = 1:length(subsegment)
	line([subsegment(ii,1) ; subsegment(ii,3)], [subsegment(ii,2) ; subsegment(ii,4)], 'Color', 'blue');
end;
