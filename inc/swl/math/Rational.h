#if !defined(__SWL_MATH__RATIONAL__H_)
#define __SWL_MATH__RATIONAL__H_ 1


#include "swl/math/ExportMath.h"
#include "swl/math/MathConstant.h"
#include <iostream>
#include <deque>


namespace swl {

//-----------------------------------------------------------------------------------------
// class Rational: rational number

class SWL_MATH_API Rational
{
public:
	Rational();
	explicit Rational(int i);
	explicit Rational(long l);
	explicit Rational(float f);
	explicit Rational(double d);
	explicit Rational(int num, int denom);
	explicit Rational(long num, long denom);
	Rational(const Rational& rhs);
	~Rational();
	
	Rational& operator=(const Rational& rhs);

public:
	/// accessor
	long num() const  {  return long(num_ * sign_);  }
	unsigned long denom() const  {  return denom_;  }

	/// mutator
	void set(int num, int denom);
	void set(long num, long denom);
	void set(double num, double denom, double tolerance = MathConstant::EPS);
	void set(const Rational& r);

	///
	float toFloat() const;
	double toDouble() const;

	///
	bool isZero(double tolerance = MathConstant::EPS) const;
	bool isEqual(const Rational& rRational, double tolerance = MathConstant::EPS) const;
	
	/// comparison operator
	bool operator==(const Rational& rhs) const;
	bool operator!=(const Rational& rhs) const;
	bool operator> (const Rational& rhs) const;
	bool operator>=(const Rational& rhs) const;
	bool operator< (const Rational& rhs) const;
	bool operator<=(const Rational& rhs) const;

	/// arithmetic operation
	Rational& operator+();
	Rational operator+(const Rational& rhs) const;
	Rational& operator+=(const Rational& rhs);
	Rational operator-() const;
	Rational operator-(const Rational& rhs) const;
	Rational& operator-=(const Rational& rhs);
	Rational operator*(const Rational& rhs) const;
	Rational& operator*=(const Rational& rhs);
	Rational operator/(const Rational& rhs) const;
	Rational& operator/=(const Rational& rhs);
/*
	friend Rational operator+(int i, Rational& rhs);
	friend Rational operator-(int i, Rational& rhs);
	friend Rational operator*(int i, Rational& rhs);
	friend Rational operator/(int i, Rational& rhs);
	friend Rational operator+(long l, Rational& rhs);
	friend Rational operator-(long l, Rational& rhs);
	friend Rational operator*(long l, Rational& rhs);
	friend Rational operator/(long l, Rational& rhs);
	friend Rational operator+(float f, Rational& rhs);
	friend Rational operator-(float f, Rational& rhs);
	friend Rational operator*(float f, Rational& rhs);
	friend Rational operator/(float f, Rational& rhs);
	friend Rational operator+(double d, Rational& rhs);
	friend Rational operator-(double d, Rational& rhs);
	friend Rational operator*(double d, Rational& rhs);
	friend Rational operator/(double d, Rational& rhs);
*/	
	/// static function
	static Rational toRational(double r, double tolerance = MathConstant::EPS);

private:
	///
	static bool isOverflow(double value);

	///
	void makeMinimalRational();

	///
	static void calcContinuousRational(const std::deque<long>& denomCtr, Rational& r);

private:
	/// signum
	int sign_;

	/// numerator
	unsigned long num_;
	/// denominator
	unsigned long denom_;
};


//-----------------------------------------------------------------------------------------
// Rational Number API

///  stream function
SWL_MATH_API std::istream& operator>>(std::istream& stream, Rational& r);
SWL_MATH_API std::ostream& operator<<(std::ostream& stream, const Rational& r);

///  analytic function
SWL_MATH_API Rational sqrt(const Rational& r, double tolerance = MathConstant::EPS);
SWL_MATH_API Rational pow(const Rational& r1, const Rational& r2, double tolerance = MathConstant::EPS);
//SWL_MATH_API Rational pow10(const Rational& r, double tolerance = MathConstant::EPS);
SWL_MATH_API Rational exp(const Rational& r, double tolerance = MathConstant::EPS);

SWL_MATH_API Rational ln(const Rational& r, double tolerance = MathConstant::EPS);
SWL_MATH_API Rational log(const Rational& r, double tolerance = MathConstant::EPS);
SWL_MATH_API Rational log10(const Rational& r, double tolerance = MathConstant::EPS);
SWL_MATH_API Rational logb(double base, const Rational& r, double tolerance = MathConstant::EPS);

SWL_MATH_API Rational sin(const Rational& r, double tolerance = MathConstant::EPS);
SWL_MATH_API Rational cos(const Rational& r, double tolerance = MathConstant::EPS);
SWL_MATH_API Rational tan(const Rational& r, double tolerance = MathConstant::EPS);
SWL_MATH_API Rational asin(const Rational& r, double tolerance = MathConstant::EPS);
SWL_MATH_API Rational acos(const Rational& r, double tolerance = MathConstant::EPS);
SWL_MATH_API Rational atan(const Rational& r, double tolerance = MathConstant::EPS);
SWL_MATH_API Rational atan2(const Rational& r1, const Rational& r2, double tolerance = MathConstant::EPS);

SWL_MATH_API Rational sinh(const Rational& r, double tolerance = MathConstant::EPS);
SWL_MATH_API Rational cosh(const Rational& r, double tolerance = MathConstant::EPS);
SWL_MATH_API Rational tanh(const Rational& r, double tolerance = MathConstant::EPS);
SWL_MATH_API Rational asinh(const Rational& r, double tolerance = MathConstant::EPS);
SWL_MATH_API Rational acosh(const Rational& r, double tolerance = MathConstant::EPS);
SWL_MATH_API Rational atanh(const Rational& r, double tolerance = MathConstant::EPS);

}  // namespace swl


#endif  // __SWL_MATH__RATIONAL__H_
