function prob = movmf_pdf_2d(theta, phi, mu, kappa, alpha)

% x: a unit direction vector on 2-dimensional sphere, norm(x) = 1, column-major vector.
x = [ sin(theta)*cos(phi) ; sin(theta)*sin(phi) ; cos(theta) ];
prob = movmf_pdf(x, mu, kappa, alpha);
