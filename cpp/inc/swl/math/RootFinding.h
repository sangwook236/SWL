#if !defined(__SWL_MATH__ROOT_FINDING__H_)
#define __SWL_MATH__ROOT_FINDING__H_ 1


#include "swl/math/MathConstant.h"
#include "swl/math/Complex.h"
#include <vector>
#include <array>
#include <complex>


namespace swl {

//-----------------------------------------------------------------------------------------
// Root Finding.

struct SWL_MATH_API RootFinding
{
public:
	/// Solve f(x) = 0.
	static double secant(double init, double (*func)(double), double tol = MathConstant::EPS);
	//template<class FO> static double secant(double init, FO foFunc, double tol = MathConstant::EPS);
	static double bisection(double left, double right, double (*func)(double), double tol = MathConstant::EPS);
	//template<class FO> static double bisection(double left, double right, FO foFunc, double tol = MathConstant::EPS);
	static double falsePosition(double left, double right, double (*func)(double), double tol = MathConstant::EPS);
	//template<class FO> static double falsePosition(double left, double right, FO foFunc, double tol = MathConstant::EPS);

	/// Solve polynomial.
	static bool quadratic(const std::array<double, 3>& coeffs, std::array<std::complex<double>, 2>& roots, double tol = MathConstant::EPS);
	static bool cubic(const std::array<double, 4>& coeffs, std::array<std::complex<double>, 3>& roots, double tol = MathConstant::EPS);
	static bool quartic(const std::array<double, 5>& coeffs, std::array<std::complex<double>, 4>& roots, double tol = MathConstant::EPS);

	static bool bairstow(const std::vector<double>& coeffs, std::vector<std::complex<double> >& roots, double tol = MathConstant::EPS);

	/// Solve a system of polynomial equations.
	/// Solve a system of two quadratic equations.
	/// Quadratic equations: a1 * x^2 + b1 * x + c1 * y + d1 = 0 & a2 * x^2 + b2 * x + c2 * y + d2 = 0.
	/// @return -1 if infinitely many real solutions exist. 0, 1, 2 as the number of real solutions.
	static size_t solveSystemOfQuadraticEquations(const double a1, const double b1, const double c1, const double d1, const double a2, const double b2, const double c2, const double d2, double &x0, double& x1, double tol = MathConstant::TOL_5);
};

}  // namespace swl


#endif  // __SWL_MATH__ROOT_FINDING__H_
