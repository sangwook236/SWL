function angles = HoG_to_angle(HoG, HoG_bin_width, HoG_scale_factor)

% covert histogram of gradients (HoG) into 1-dimensional direction angles
%
% HoG: histogram of gradients.
% if HoG_bin_width = 1 (1 deg), [0 360) = ( 0, 1, 2, ..., 359 } [deg].
% if HoG_bin_width = 10 (10 deg), [0 360) = ( 0, 10, 20, ..., 350 } [deg].
% angles: 1-dimensional direction angles.

% directions: 2-dimensional direction vectors, row-major vector.

histo = round(HoG * HoG_scale_factor);
angles = zeros(sum(histo), 1);
%directions = zeros(sum(histo), 2);  % row-major

numBins = length(histo);
idx = 1;
for ii = 1:numBins
    for jj = 1:histo(ii)
        angles(idx) = (ii - 1) * HoG_bin_width * pi / 180;
        %directions(idx, :) = [ cos(ang) sin(ang) ];
        idx = idx + 1;
    end;
end;
