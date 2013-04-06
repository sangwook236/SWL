T = 100;  % [sec]
L = [ 10 10 10 ];  % [m]

% 6-th order polynomial
% p(t) = a6 * t^6 + a5 * t^5 + a4 * t^4 + a3 * t^3 + a2 * t^2 + a1 * t + a0

%------------------------------------------------------------------------------

% pre-defined coefficient
predefined_coeffs = [ 0.002 -0.003 -0.015 ];  % [ a4 a4 a4 ]

A = [
	T^6    T^5    T^3
	6*T^5  5*T^4  3*T^2
	30*T^4 20*T^3 6*T
];
for ii = 1:3
	B(:,ii) = [ L(ii)-predefined_coeffs(ii)*T^4 ; -4*predefined_coeffs(ii)*T^3 ; -12*predefined_coeffs(ii)*T^2 ];
end;

% coefficients: [ a6 a5 a3 ]
coeff = inv(A) * B;

time = [0:0.1:T]';
for ii = 1:3
	poly_coeffs(:,ii) = [ coeff(1,ii) coeff(2,ii) predefined_coeffs(ii) coeff(3,ii) 0 0 0 ]';
	traj_vals(:,ii) = polyval(poly_coeffs(:,ii), time);
end;

figure;
plot(time, traj_vals(:,1), 'r-', time, traj_vals(:,2), 'g-', time, traj_vals(:,3), 'b-');

%------------------------------------------------------------------------------

% pre-defined coefficient
predefined_coeffs = [ 0.002 -0.003 -0.015 ];  % [ a5 a3 a4 ]

AA = [
	T^6    T^5    T^4    T^3
	6*T^5  5*T^4  4*T^3  3*T^2
	30*T^4 20*T^3 12*T^2 6*T
];
BB(:,1) = [ L(1)-predefined_coeffs(1)*T^5 ; -5*predefined_coeffs(1)*T^4 ; -20*predefined_coeffs(1)*T^3 ];
BB(:,2) = [ L(2)-predefined_coeffs(2)*T^3 ; -3*predefined_coeffs(2)*T^2 ; -6*predefined_coeffs(2)*T ];
BB(:,3) = [ L(3)-predefined_coeffs(3)*T^4 ; -4*predefined_coeffs(3)*T^3 ; -12*predefined_coeffs(3)*T^2 ];

% coefficients
coeff(:,1) = inv(AA(:,[1 3 4])) * BB(:,1);  % use a6, a4, & a3
coeff(:,2) = inv(AA(:,[1 2 3])) * BB(:,2);  % use a6, a5, & a4
coeff(:,3) = inv(AA(:,[1 2 4])) * BB(:,3);  % use a6, a5, & a3

poly_coeffs(:,1) = [ coeff(1,1) predefined_coeffs(1) coeff(2,1) coeff(3,1) 0 0 0 ]';
poly_coeffs(:,2) = [ coeff(1,2) coeff(2,2) coeff(3,2) predefined_coeffs(2) 0 0 0 ]';
poly_coeffs(:,3) = [ coeff(1,3) coeff(2,3) predefined_coeffs(3) coeff(3,3) 0 0 0 ]';

time = [0:0.1:T]';
for ii = 1:3
	traj_vals(:,ii) = polyval(poly_coeffs(:,ii), time);
end;

figure;
plot(time, traj_vals(:,1), 'r-', time, traj_vals(:,2), 'g-', time, traj_vals(:,3), 'b-');
