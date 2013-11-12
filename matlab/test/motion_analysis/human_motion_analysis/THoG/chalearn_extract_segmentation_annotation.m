function [ annotation ] = chalearn_extract_segmentation_annotation(segmentation_annotation_file_path)

% [ dataset index, video file index, gesture index, gesture label ID, start frame, end frame ].

fid = fopen(segmentation_annotation_file_path);
annotation_info = textscan(fid, '%s%d%d%d%d%d', 'Delimiter', ',', 'Headerlines', 1);
fclose(fid);

%dataset_name,videos,gestures,labels,Start,End
%devel01,     13,    1,       4,     4,    35
%devel01,     13,    2,       8,     40,   62

annotation_len = length(annotation_info{1});
annotation = zeros(annotation_len, 6);
for ii = 1:annotation_len
	% dataset index.
	annotation(ii,1) = str2num(annotation_info{1}{ii}(6:7));  % e.g.) devel01.
	% video file index.
	annotation(ii,2) = annotation_info{2}(ii);
	% gesture index.
	annotation(ii,3) = annotation_info{3}(ii);
	% gesture label ID.
	annotation(ii,4) = annotation_info{4}(ii);
	% start frame.
	annotation(ii,5) = annotation_info{5}(ii);
	% end frame.
	annotation(ii,6) = annotation_info{6}(ii);
end;
