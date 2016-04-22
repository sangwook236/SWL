#if !defined(__SWL_MATH__MATH_UTIL__H_)
#define __SWL_MATH__MATH_UTIL__H_ 1


#include "swl/math/MathConstant.h"
#include <string>


namespace swl {

//-----------------------------------------------------------------------------------------
// struct MathUtil

struct SWL_MATH_API MathUtil
{
public:
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

	/// this function wraps a value, x in the interval [lower, upper)
	static double wrap(const double x, const double lower, const double upper, const double tol = MathConstant::EPS);

	///
	static bool isInteger(const double x, const double tol = MathConstant::EPS);
	static bool isReal(const double x, const double tol = MathConstant::EPS);

	///
	static double toRad(const double deg)
	{  return deg * MathConstant::TO_RAD;  }
	static double toDeg(const double rad)
	{  return rad * MathConstant::TO_DEG;  }

	/// a number to an odd number.
	static int toPrecedingOdd(const double num);
	static int toFollowingOdd(const double num);
	/// a number to an even number.
	static int toPrecedingEven(const double num);
	static int toFollowingEven(const double num);

	/// convert base field
#if defined(_UNICODE) || defined(UNICODE)
	static std::wstring dec2bin(const long dec);
	static std::wstring dec2oct(const long dec);
	static std::wstring dec2hex(const long dec);

	static long bin2dec(const std::wstring &bin);
	static long oct2dec(const std::wstring &oct);
	static long hex2dec(const std::wstring &hex);
#else
	static std::string dec2bin(const long dec);
	static std::string dec2oct(const long dec);
	static std::string dec2hex(const long dec);

	static long bin2dec(const std::string &bin);
	static long oct2dec(const std::string &oct);
	static long hex2dec(const std::string &hex);
#endif
};

}  // namespace swl


#endif  // __SWL_MATH__MATH_UTIL__H_
