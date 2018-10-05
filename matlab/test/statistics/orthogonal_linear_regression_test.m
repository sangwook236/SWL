%addpath('D:/work/SWL_github/matlab/src/statistics');

x = [0:0.1:10]';
y = 2 + 7 * x + 6 * randn(size(x));

coeffs1 = orthogonal_linear_regression(x, y);
y1 = -(coeffs1(1) * x + coeffs1(3)) / coeffs1(2);

coeffs2 = linear_regression(x, y);
y2 = -(coeffs2(1) * x + coeffs2(3)) / coeffs2(2);

coeffs3 = polyfit(x, y, 1);  % Linear regression. (?)
y3 = polyval(coeffs3, x);

plot(x, y, '.', x, y1, 'r-', x, y2, 'g-', x, y3, 'b:');
