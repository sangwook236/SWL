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
	explicit Rational(const int i);
	explicit Rational(const long l);
	explicit Rational(const float f);
	explicit Rational(const double d);
	explicit Rational(const int num, const int denom);
	explicit Rational(const long num, const long denom);
	Rational(const Rational &rhs);
	~Rational();
	
	Rational& operator=(const Rational &rhs);

public:
	/// accessor
	long num() const  {  return long(num_ * sign_);  }
	unsigned long denom() const  {  return denom_;  }

	/// mutator
	void set(const int num, const int denom);
	void set(const long num, const long denom);
	void set(const double num, const double denom, const double tolerance = MathConstant::EPS);
	void set(const Rational &r);

	///
	float toFloat() const;
	double toDouble() const;

	///
	bool isZero(const double tolerance = MathConstant::EPS) const;
	bool isEqual(const Rational &r, const double tolerance = MathConstant::EPS) const;
	
	/// comparison operator
	bool operator==(const Rational &rhs) const;
	bool operator!=(const Rational &rhs) const;
	bool operator> (const Rational &rhs) const;
	bool operator>=(const Rational &rhs) const;
	bool operator< (const Rational &rhs) const;
	bool operator<=(const Rational &rhs) const;

	/// arithmetic operation
	Rational & operator+();
	Rational operator+(const Rational &rhs) const;
	Rational & operator+=(const Rational &rhs);
	Rational operator-() const;
	Rational operator-(const Rational &rhs) const;
	Rational & operator-=(const Rational &rhs);
	Rational operator*(const Rational &rhs) const;
	Rational & operator*=(const Rational &rhs);
	Rational operator/(const Rational &rhs) const;
	Rational & operator/=(const Rational &rhs);
/*
	friend Rational operator+(const int i, const Rational &rhs);
	friend Rational operator-(const int i, const Rational &rhs);
	friend Rational operator*(const int i, const Rational &rhs);
	friend Rational operator/(const int i, const Rational &rhs);
	friend Rational operator+(const long l, const Rational &rhs);
	friend Rational operator-(const long l, const Rational &rhs);
	friend Rational operator*(const long l, const Rational &rhs);
	friend Rational operator/(const long l, const Rational &rhs);
	friend Rational operator+(const float f, const Rational &rhs);
	friend Rational operator-(const float f, const Rational &rhs);
	friend Rational operator*(const float f, const Rational &rhs);
	friend Rational operator/(const float f, const Rational &rhs);
	friend Rational operator+(const double d, const Rational &rhs);
	friend Rational operator-(const double d, const Rational &rhs);
	friend Rational operator*(const double d, const Rational &rhs);
	friend Rational operator/(const double d, const Rational &rhs);
*/	
	/// static function
	static Rational toRational(const double r, const double tolerance = MathConstant::EPS);

private:
	///
	static bool isOverflow(const double value);

	///
	void makeMinimalRational();

	///
	static void calcContinuousRational(const std::deque<long> &denomCtr, Rational &r);

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
SWL_MATH_API std::istream & operator>>(std::istream &stream, Rational &r);
SWL_MATH_API std::ostream & operator<<(std::ostream &stream, const Rational &r);

///  analytic function
SWL_MATH_API Rational sqrt(const Rational &r, const double tolerance = MathConstant::EPS);
SWL_MATH_API Rational pow(const Rational &r1, const Rational &r2, const double tolerance = MathConstant::EPS);
//SWL_MATH_API Rational pow10(const Rational &r, const double tolerance = MathConstant::EPS);
SWL_MATH_API Rational exp(const Rational &r, const double tolerance = MathConstant::EPS);

SWL_MATH_API Rational ln(const Rational &r, const double tolerance = MathConstant::EPS);
SWL_MATH_API Rational log(const Rational &r, const double tolerance = MathConstant::EPS);
SWL_MATH_API Rational log10(const Rational &r, const double tolerance = MathConstant::EPS);
SWL_MATH_API Rational logb(const double base, const Rational &r, const double tolerance = MathConstant::EPS);

SWL_MATH_API Rational sin(const Rational &r, const double tolerance = MathConstant::EPS);
SWL_MATH_API Rational cos(const Rational &r, const double tolerance = MathConstant::EPS);
SWL_MATH_API Rational tan(const Rational &r, const double tolerance = MathConstant::EPS);
SWL_MATH_API Rational asin(const Rational &r, const double tolerance = MathConstant::EPS);
SWL_MATH_API Rational acos(const Rational &r, const double tolerance = MathConstant::EPS);
SWL_MATH_API Rational atan(const Rational &r, const double tolerance = MathConstant::EPS);
SWL_MATH_API Rational atan2(const Rational &r1, const Rational &r2, const double tolerance = MathConstant::EPS);

SWL_MATH_API Rational sinh(const Rational &r, const double tolerance = MathConstant::EPS);
SWL_MATH_API Rational cosh(const Rational &r, const double tolerance = MathConstant::EPS);
SWL_MATH_API Rational tanh(const Rational &r, const double tolerance = MathConstant::EPS);
SWL_MATH_API Rational asinh(const Rational &r, const double tolerance = MathConstant::EPS);
SWL_MATH_API Rational acosh(const Rational &r, const double tolerance = MathConstant::EPS);
SWL_MATH_API Rational atanh(const Rational &r, const double tolerance = MathConstant::EPS);

}  // namespace swl


#endif  // __SWL_MATH__RATIONAL__H_
