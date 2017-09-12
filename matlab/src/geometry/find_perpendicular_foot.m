function intersection_pt = find_perpendicular_foot(line, pt)
% Find a perpendicular foot from a point to a line in 2-dimensional space.
% line: 3x1 vector [l1 l2 l3]' (possibly l3 = 1) -> line equation: l1 * x + l2 * y + l3 * z = 0.
% pt: 3x1 vector [x y z]' (z ~= 0, possibly z = 1).

a = (line(1) * pt(2) - line(2) * pt(1)) / pt(3);
intersection_pt = cross(line, [line(2) ; -line(1) ; a]);  % The intersection point of two lines.
intersection_pt = intersection_pt / intersection_pt(3);
