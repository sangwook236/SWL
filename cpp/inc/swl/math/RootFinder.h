#if !defined(__SWL_MATH__ROOT_FINDER__H_)
#define __SWL_MATH__ROOT_FINDER__H_ 1


#include "swl/math/MathConstant.h"
#include "swl/math/Complex.h"
#include <vector>


namespace swl {

//-----------------------------------------------------------------------------------------
// struct RootFinder

struct SWL_MATH_API RootFinder
{
public:
	/// solve f(x) = 0
	static double secant(double init, double (*func)(double), double tolerance = MathConstant::EPS);
	//template<class FO> static double secant(double init, FO foFunc, double tolerance = MathConstant::EPS);
	static double bisection(double left, double right, double (*func)(double), double tolerance = MathConstant::EPS);
	//template<class FO> static double bisection(double left, double right, FO foFunc, double tolerance = MathConstant::EPS);
	static double falsePosition(double left, double right, double (*func)(double), double tolerance = MathConstant::EPS);
	//template<class FO> static double falsePosition(double left, double right, FO foFunc, double tolerance = MathConstant::EPS);

	/// solve polynomial
	static bool quadratic(const double coeffArr[3], Complex<double> rootArr[2], double tolerance = MathConstant::EPS);
	static bool quadratic(const std::vector<double>& coeffCtr, std::vector<Complex<double> >& rootCtr, double tolerance = MathConstant::EPS);
	static bool cubic(const double coeffArr[4], Complex<double> rootArr[3], double tolerance = MathConstant::EPS);
	static bool cubic(const std::vector<double>& coeffCtr, std::vector<Complex<double> >& rootCtr, double tolerance = MathConstant::EPS);
	static bool quartic(const double coeffArr[5], Complex<double> rootArr[4], double tolerance = MathConstant::EPS);
	static bool quartic(const std::vector<double>& coeffCtr, std::vector<Complex<double> >& rootCtr, double tolerance = MathConstant::EPS);

	static bool bairstow(const std::vector<double>& coeffCtr, std::vector<Complex<double> >& rootCtr, double tolerance = MathConstant::EPS);
};

}  // namespace swl


#endif  // __SWL_MATH__ROOT_FINDER__H_
