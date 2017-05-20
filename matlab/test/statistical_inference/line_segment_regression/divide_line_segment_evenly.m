function subsegment = divide_line_segment_evenly(ls, ref_len)
% A finite line segment: ls = [ x1 y1 x2 y2 ].
%	(x1, y1) - (x2, y2).
% ref_len: the reference length for sub-segments.

segment_len = sqrt((ls(:,3) - ls(:,1)).^2 + (ls(:,4) - ls(:,2)).^2);
segment_dir = [ ls(:,3) - ls(:,1) ls(:,4) - ls(:,2) ] ./ segment_len;
subsegment_count = ceil(segment_len / ref_len);
subsegment_len = segment_len ./ subsegment_count;

subsegment = zeros(sum(subsegment_count), 4);
idx = 1;
for ii = 1:length(subsegment_len)
	pt1 = [ ls(ii,1) ls(ii,2) ];
	for jj = 1:subsegment_count(ii)
		pt2 = pt1 + segment_dir(ii,:);
		subsegment(idx,:) = [ pt1 pt2 ];
		pt1 = pt2;
		idx = idx + 1;
	end;
end;

end;
