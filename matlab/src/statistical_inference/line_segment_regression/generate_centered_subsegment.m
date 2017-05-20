function subsegment = generate_centered_subsegment(ls, ref_len)
% Generate subsegments from a finite line segment ls, which are centered at the center of ls and are of the same length.
%
% A finite line segment: ls = [ x1 y1 x2 y2 ].
%	(x1, y1) - (x2, y2).
% ref_len: the reference length for sub-segments.
%
% subsegment: A set of generated subsegments.

segment_len = sqrt((ls(:,3) - ls(:,1)).^2 + (ls(:,4) - ls(:,2)).^2);
segment_dir = [ (ls(:,3) - ls(:,1)) ./ segment_len (ls(:,4) - ls(:,2)) ./ segment_len ];
subsegment_count = ceil(segment_len / ref_len);
subsegment_len = segment_len ./ subsegment_count;

xc = (ls(:,1) + ls(:,3)) / 2;
yc = (ls(:,2) + ls(:,4)) / 2;

subsegment = zeros(sum(subsegment_count), 4);
idx = 1;
for ii = 1:length(subsegment_len)
	pt1 = [ xc(ii) yc(ii) ] - subsegment_len(ii) * segment_dir(ii,:) / 2;
	pt2 = [ xc(ii) yc(ii) ] + subsegment_len(ii) * segment_dir(ii,:) / 2;

	% FIXME [improve] >> Vectorize.
	for jj = 1:subsegment_count(ii)
		subsegment(idx,:) = [ pt1 pt2 ];
		idx = idx + 1;
	end;
end;
