#if !defined(__SWL_MATH__CURVE_FITTING__H_)
#define __SWL_MATH__CURVE_FITTING__H_ 1


#include "swl/math/ExportMath.h"
#include "swl/base/Point.h"
#include <list>
#include <array>


namespace swl {

//-----------------------------------------------------------------------------------------
// Curve Fitting.

struct SWL_MATH_API CurveFitting
{
public:
	/// Line equation: a * x + b * y + c = 0.
	static bool estimateLineByLeastSquares(const std::list<std::array<double, 2> >& points, double& a, double& b, double& c);
	/// Quadratic equation: a * x^2 + b * x + c * y + d = 0.
	/// Line equation: b * x + c * y + d = 0 if a = 0.
	static bool estimateQuadraticByLeastSquares(const std::list<std::array<double, 2> >& points, double& a, double& b, double& c, double& d);
};

}  // namespace swl


#endif  // __SWL_MATH__CURVE_FITTING__H_
