#if !defined(__SWL_MATH__MATH_EXTENSION__H_)
#define __SWL_MATH__MATH_EXTENSION__H_ 1


#include "swl/math/MathConstant.h"


namespace swl {

//-----------------------------------------------------------------------------------------
// struct MathExt

struct SWL_MATH_API MathExt
{
public:
	///
	static double mantissa(const double x, double *pow = 0L);
	static double saturate(const double x)
	{  return x >= 1.0 ? 1.0 : (x <= -1.0 ? -1.0 : x);  }

	///
	static double round(const double x);
	double round(const double x, const int decimalPlace);

	///
	static double logb(double base, const double x);
	static double asinh(const double x);
	/// x >= 1
	static double acosh(const double x);
	/// -1 < x < 1
	static double atanh(const double x);

	/// GCD: greatest common divisor
	static unsigned long gcd(const unsigned long lhs, const unsigned long rhs);
	static double gcd(const double lhs, const double rhs, const double tol = MathConstant::EPS);

	/// LCM: least common multiplier
	static unsigned long lcm(const unsigned long lhs, const unsigned long rhs);
	static double lcm(const double lhs, const double rhs, const double tol = MathConstant::EPS);

	///
	static unsigned long factorial(const unsigned long n);
	static double factorial(const double n, const double tol = MathConstant::EPS);
	static unsigned long permutation(const unsigned long lhs, const unsigned long rhs);
	static double permutation(const double lhs, const double rhs, const double tol = MathConstant::EPS);
	static unsigned long binomial(const unsigned long lhs, const unsigned long rhs);
	static double binomial(const double lhs, const double rhs, const double tol = MathConstant::EPS);
};

}  // namespace swl


#endif  //  __SWL_MATH__MATH_EXTENSION__H_
