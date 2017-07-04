function [ subsegment subsegment_weight ] = generate_subsegment_randomly(ls, ref_len)
% Generate subsegments at random from a finite line segment ls.
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

subsegment = zeros(sum(subsegment_count), 4);
subsegment_weight = zeros(sum(subsegment_count), 1);
idx = 1;
for ii = 1:length(subsegment_len)
	xc = ls(ii,1) + (ls(ii,3) - ls(ii,1)) .* rand([subsegment_count(ii), 1]);
	if ls(ii,3) == ls(ii,1)
		yc = ls(ii,2) * ones(size(xc));
	else
		yc = ((ls(ii,4) - ls(ii,2)) / (ls(ii,3) - ls(ii,1))) * (xc - ls(ii,1)) + ls(ii,2);
	end;

	% FIXME [improve] >> Vectorize.
	for jj = 1:subsegment_count(ii)
		pt1 = [ xc(jj) yc(jj) ] - subsegment_len(ii) * segment_dir(ii,:) / 2;
		pt2 = [ xc(jj) yc(jj) ] + subsegment_len(ii) * segment_dir(ii,:) / 2;
		subsegment(idx,:) = [ pt1 pt2 ];
		subsegment_weight(idx) = subsegment_len(ii) / ref_len;
		idx = idx + 1;
	end;
end;
