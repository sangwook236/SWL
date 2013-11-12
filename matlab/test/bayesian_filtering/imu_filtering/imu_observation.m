function y_n = imu_observation(x_n, param)

state_dim = 22;
output_dim = 6;

Px = x_n(1);
Py = x_n(2);
Pz = x_n(3);
Vx = x_n(4);
Vy = x_n(5);
Vz = x_n(6);
Ap = x_n(7);
Aq = x_n(8);
Ar = x_n(9);
E0 = x_n(10);
E1 = x_n(11);
E2 = x_n(12);
E3 = x_n(13);
Wp = x_n(14);
Wq = x_n(15);
Wr = x_n(16);
Abp = x_n(17);
Abq = x_n(18);
Abr = x_n(19);
Wbp = x_n(20);
Wbq = x_n(21);
Wbr = x_n(22);

y_n = zeros(output_dim,1);
y_n(1) = Ap + Abp;
y_n(2) = Aq + Abq;
y_n(3) = Ar + Abr;
y_n(4) = Wp + Wbp;
y_n(5) = Wq + Wbq;
y_n(6) = Wr + Wbr;

if size(x_n,1) > 2*state_dim
	for ii = 1:output_dim
		y_n(ii) = y_n(ii) + x_n(2*state_dim+ii,:);
	end;
end
