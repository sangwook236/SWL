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
	/**
	 *	@defgroup SolveEquation Solve an equation, f(x) = 0
	 *	@{
	 */

	/**
	 *	@brief Solve f(x) = 0 using the secant method.
	 */
	static double secant(double init, double (*func)(double), const double& tol = MathConstant::EPS);
	//template<class Functor> static double secant(double init, Functor func, const double& tol = MathConstant::EPS);
	/**
	 *	@brief Solve f(x) = 0 using the bisection method.
	 */
	static double bisection(double left, double right, double (*func)(double), const double& tol = MathConstant::EPS);
	//template<class Functor> static double bisection(double left, double right, Functor func, const double& tol = MathConstant::EPS);
	/**
	 *	@brief Solve f(x) = 0 using the false position method.
	 */
	static double falsePosition(double left, double right, double (*func)(double), const double& tol = MathConstant::EPS);
	//template<class Functor> static double falsePosition(double left, double right, Functor func, const double& tol = MathConstant::EPS);

	/**
	 *	@}
	 */

	/**
	 *	@defgroup SolvePolynomial Solve a polynomial equation
	 *	@{
	 */

	/**
	 *	@brief Solve 2nd order polynomials.
	 *	Coefficients are arranged by a descending order.
	 */
	static bool quadratic(const std::array<double, 3>& coeffs, std::array<std::complex<double>, 2>& roots, const double& tol = MathConstant::EPS);
	/**
	 *	@brief Solve 3rd order polynomials.
	 *	Coefficients are arranged by a descending order.
	 */
	static bool cubic(const std::array<double, 4>& coeffs, std::array<std::complex<double>, 3>& roots, const double& tol = MathConstant::EPS);
	/**
	 *	@brief Solve 4th order polynomials.
	 *	Coefficients are arranged by a descending order.
	 */
	static bool quartic(const std::array<double, 5>& coeffs, std::array<std::complex<double>, 4>& roots, const double& tol = MathConstant::EPS);

	/**
	 *	@brief Solve n-th order polynomials using Bairstow's method.
	 *	Coefficients are arranged by a descending order.
	 */
	static bool bairstow(const std::vector<double>& coeffs, std::vector<std::complex<double> >& roots, const double& eps = MathConstant::EPS);

	/**
	 *	@}
	 */

	/**
	 *	@defgroup SolveSystemOfPolynomials Solve a system of polynomial equations
	 *	@{
	 */

	/**
	 *	@brief Solve a system of two quadratic equations.
	 *	Quadratic equations: a1 * x^2 + b1 * x + c1 * y + d1 = 0 & a2 * x^2 + b2 * x + c2 * y + d2 = 0. <br/>
	 *	(a1, b1, c1, d1) = coeffs1, (a2, b2, c2, d2) = coeffs2.
	 *	@return -1 if infinitely many real solutions exist. 0, 1, 2 as the number of real solutions.
	 */
	static size_t solveSystemOfQuadraticEquations(const std::array<double, 4>& coeffs1, const std::array<double, 4>& coeffs2, std::array<double, 2>& roots, const double& tol = MathConstant::TOL_5);

	/**
	 *	@}
	 */
};

}  // namespace swl


#endif  // __SWL_MATH__ROOT_FINDING__H_
