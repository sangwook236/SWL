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

	static int sign(const double x)
	{  return isZero(x) ? 0 : (x > 0.0 ? 1 : -1);  }
	static bool isPositive(const double x, const double tol = MathConstant::EPS)
	{  return x > tol;  }
	static bool isNegative(const double x, const double tol = MathConstant::EPS)
	{  return x < -tol;  }

	/// check if a value is bouned in the interval (lower, upper)
	static bool isBounded_oo(const double x, const double lower, const double upper, const double tol = MathConstant::EPS)
	{  return lower - tol < x && x < upper + tol;  }
	/// check if a value is bouned in the interval [lower, upper]
	static bool isBounded_cc(const double x, const double lower, const double upper, const double tol = MathConstant::EPS)
	{  return lower - tol <= x && x <= upper + tol;  }
	/// check if a value is bouned in the interval (lower, upper]
	static bool isBounded_oc(const double x, const double lower, const double upper, const double tol = MathConstant::EPS)
	{  return lower - tol < x && x <= upper + tol;  }
	/// check if a value is bouned in the interval [lower, upper)
	static bool isBounded_co(const double x, const double lower, const double upper, const double tol = MathConstant::EPS)
	{  return lower - tol <= x && x < upper + tol;  }

	///
	static double toRad(const double deg)
	{  return deg * MathConstant::TO_RAD;  }
	static double toDeg(const double rad)
	{  return rad * MathConstant::TO_DEG;  }

	///
	static bool isInteger(const double x, const double tol = MathConstant::EPS);
	static bool isReal(const double x, const double tol = MathConstant::EPS);

	/// convert base field
#if defined(_UNICODE) || defined(UNICODE)
	std::wstring toBin(const unsigned long dec);
	std::wstring toOct(const unsigned long dec);
	std::wstring toHex(const unsigned long dec);
#else
	std::string toBin(const unsigned long dec);
	std::string toOct(const unsigned long dec);
	std::string toHex(const unsigned long dec);
#endif
};

}  // namespace swl


#endif  // __SWL_MATH__MATH_UTILITY__H_
