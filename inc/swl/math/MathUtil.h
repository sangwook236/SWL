#if !defined(__SWL_MATH__MATH_UTILITY__H_)
#define __SWL_MATH__MATH_UTILITY__H_ 1


#include "swl/math/MathConstant.h"
#include <string>


namespace swl {

//-----------------------------------------------------------------------------------------
// struct MathUtil

struct SWL_MATH_API MathUtil
{
public :
	///
	static bool isZero(const double x, const double tol = MathConstant::EPS)
	{  return -tol <= x && x <= tol;  }
	static bool isNotZero(const double x, const double tol = MathConstant::EPS)
	{  return x < -tol || x > tol;  }

	static bool isEqual(const double x, const double y, const double tol = MathConstant::EPS)
	{  return isZero(x - y, tol);  }

	static double sign(const double x)
	{  return x >= 0.0 ? 1.0 : -1.0;  }
	static bool isPositive(const double x, const double tol = MathConstant::EPS)
	{  return x > tol;  }
	static bool isNegative(const double x, const double tol = MathConstant::EPS)
	{  return x < -tol;  }

	///  (lower, upper)
	static bool isBounded_oo(double x, const double lower, const double upper, const double tol = MathConstant::EPS)
	{  return lower - tol < x && x < upper + tol;  }
	///  [lower, upper]
	static bool isBounded_cc(double x, double lower, double upper, const double tol = MathConstant::EPS)
	{  return lower - tol <= x && x <= upper + tol;  }
	///  (lower, upper]
	static bool isBounded_oc(double x, double lower, double upper, const double tol = MathConstant::EPS)
	{  return lower - tol < x && x <= upper + tol;  }
	///  [lower, upper)
	static bool isBounded_co(double x, double lower, double upper, const double tol = MathConstant::EPS)
	{  return lower - tol <= x && x < upper + tol;  }

	///
	static double toRad(double deg)
	{  return deg * MathConstant::TO_RAD;  }
	static double toDeg(double rad)
	{  return rad * MathConstant::TO_DEG;  }

	///
	static bool isInteger(const double x, const double tol = MathConstant::EPS);
	static bool isReal(const double x, const double tol = MathConstant::EPS);

	///  convert base field
	std::string toBin(const unsigned long dec);
	std::string toOct(const unsigned long dec);
	std::string toHex(const unsigned long dec);
};

}  // namespace swl


#endif  // __SWL_MATH__MATH_UTILITY__H_
