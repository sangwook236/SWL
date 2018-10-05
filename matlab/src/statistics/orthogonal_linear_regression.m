function coeffs = orthogonal_linear_regression(x, y)
% a * x + b * y + c = 0.
% coeffs = [a b c].

init = polyfit(x, y, 1);
coeffs0 = [init(2) -1 init(1)];

objfunc = inline('sum((coeffs(1) * x + coeffs(2) * y + coeffs(3)).^2) / (coeffs(1)^2 + coeffs(2)^2)', 'coeffs', 'x', 'y');
%objfunc = @(coeffs, x, y) sum((coeffs(1) * x + coeffs(2) * y + coeffs(3)).^2) / (coeffs(1)^2 + coeffs(2)^2);

options = optimoptions(@fminunc, 'Algorithm', 'quasi-newton', 'MaxIterations', 1000, 'MaxFunctionEvaluations', 1000, 'StepTolerance', 1e-6, 'OptimalityTolerance', 1e-6);
%options = optimoptions(@fminunc, 'Algorithm', 'trust-region', 'MaxIterations', 1000, 'MaxFunctionEvaluations', 1000, 'StepTolerance', 1e-6, 'OptimalityTolerance', 1e-6);
[coeffs_sol, fval, exitflag, output] = fminunc(objfunc, coeffs0, options, x, y);
%options = optimset('MaxIter', 1000, 'MaxFunEvals', 1000, 'TolX', 1e-6, 'TolFun', 1e-6);
%[coeffs_sol, fval, exitflag, output] = fminsearch(objfunc, coeffs0, options, x, y);

coeffs = coeffs_sol;
