function x_n = imu_process(x, param)

state_dim = 22;

n = param(1);
Ts = param(2);
g_ix = param(3);
g_iy = param(4);
g_iz = param(5);
beta_ax = param(6);
beta_ay = param(7);
beta_az = param(8);
beta_wx = param(9);
beta_wy = param(10);
beta_wz = param(11);

Px = x(1);
Py = x(2);
Pz = x(3);
Vx = x(4);
Vy = x(5);
Vz = x(6);
Ap = x(7);
Aq = x(8);
Ar = x(9);
E0 = x(10);
E1 = x(11);
E2 = x(12);
E3 = x(13);
Wp = x(14);
Wq = x(15);
Wr = x(16);
Abp = x(17);
Abq = x(18);
Abr = x(19);
Wbp = x(20);
Wbq = x(21);
Wbr = x(22);

w = zeros(state_dim,1);
if size(x,1) > state_dim
	for ii = 1:state_dim
		w(ii) = x(state_dim+ii,:);
	end;
end;

%dvdt_x = 2.0 * ((0.5 - E2*E2 - E3*E3)*Ap + (E1*E2 - E0*E3)*Aq + (E1*E3 + E0*E2)*Ar) + g_ix;
%dvdt_y = 2.0 * ((E1*E2 + E0*E3)*Ap + (0.5 - E1*E1 - E3*E3)*Aq + (E2*E3 - E0*E1)*Ar) + g_iy;
%dvdt_z = 2.0 * ((E1*E3 - E0*E2)*Ap + (E2*E3 + E0*E1)*Aq + (0.5 - E1*E1 - E2*E2)*Ar) + g_iz;
dvdt_x = 2.0 * ((0.5 - E2*E2 - E3*E3)*Ap + (E1*E2 - E0*E3)*Aq + (E1*E3 + E0*E2)*Ar);
dvdt_y = 2.0 * ((E1*E2 + E0*E3)*Ap + (0.5 - E1*E1 - E3*E3)*Aq + (E2*E3 - E0*E1)*Ar);
dvdt_z = 2.0 * ((E1*E3 - E0*E2)*Ap + (E2*E3 + E0*E1)*Aq + (0.5 - E1*E1 - E2*E2)*Ar);

dPhi = Wp * Ts;
dTheta = Wq * Ts;
dPsi = Wr * Ts;
s = 0.5 * sqrt(dPhi*dPhi + dTheta*dTheta + dPsi*dPsi);
lambda = 1.0 - sqrt(E0*E0 + E1*E1 + E2*E2 + E3*E3);
% TODO [check] >>
eta_dt = 0.9;  % eta * dt < 1.0

eps = 1.0e-10;
coeff1 = cos(s) + eta_dt * lambda;
if abs(s) <= eps
	coeff2 = 0.0;
else
	coeff2 = 0.5 * sin(s) / s;
end;

x_n = zeros(state_dim,1);

x_n(1) = Px + Vx * Ts;
x_n(2) = Py + Vy * Ts;
x_n(3) = Pz + Vz * Ts;

x_n(4) = Vx + dvdt_x * Ts;
x_n(5) = Vy + dvdt_y * Ts;
x_n(6) = Vz + dvdt_z * Ts;

%x_n(7) = Ap + w(7) * Ts;
%x_n(8) = Aq + w(8) * Ts;
%x_n(9) = Ar + w(9) * Ts;
exp_baxt = exp(-beta_ax * Ts);
exp_bayt = exp(-beta_ay * Ts);
exp_bazt = exp(-beta_az * Ts);
x_n(7) = Ap * exp_baxt + w(7) * (1.0 - exp_baxt);
x_n(8) = Aq * exp_bayt + w(8) * (1.0 - exp_bayt);
x_n(9) = Ar * exp_bazt + w(9) * (1.0 - exp_bazt);

x_n(10) = coeff1 * E0 - coeff2 * (dPhi*E1 + dTheta*E2 + dPsi*E3);
x_n(11) = coeff1 * E1 - coeff2 * (-dPhi*E0 - dPsi*E2 + dTheta*E3);
x_n(12) = coeff1 * E2 - coeff2 * (-dTheta*E0 + dPsi*E1 - dPhi*E3);
x_n(13) = coeff1 * E3 - coeff2 * (-dPsi*E0 - dTheta*E1 + dPhi*E2);

%x_n(14) = Wp + w(14) * Ts;
%x_n(15) = Wq + w(15) * Ts;
%x_n(16) = Wr + w(16) * Ts;
exp_bwxt = exp(-beta_wx * Ts);
exp_bwyt = exp(-beta_wy * Ts);
exp_bwzt = exp(-beta_wz * Ts);
x_n(14) = Wp * exp_bwxt + w(14) * (1.0 - exp_bwxt);
x_n(15) = Wq * exp_bwyt + w(15) * (1.0 - exp_bwyt);
x_n(16) = Wr * exp_bwzt + w(16) * (1.0 - exp_bwzt);

x_n(17) = Abp + w(17) * Ts;
x_n(18) = Abq + w(18) * Ts;
x_n(19) = Abr + w(19) * Ts;

x_n(20) = Wbp + w(20) * Ts;
x_n(21) = Wbq + w(21) * Ts;
x_n(22) = Wbr + w(22) * Ts;
