#include "swl/math/Rational.h"
#include "swl/math/MathExt.h"
#include "swl/math/MathUtil.h"
#include "swl/base/LogException.h"
#include <limits>
#include <cmath>


#if defined(_DEBUG)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif

#if defined(max)
#undef max
#endif


namespace swl {

//-----------------------------------------------------------------------------------------
// class Rational

Rational::Rational()
: sign_(1), num_(0ul), denom_(1ul)
{}

Rational::Rational(int i)
{
	if (i >= 0)
	{
		sign_ = 1;
		num_ = (unsigned long)i;
		denom_ = 1ul;
	}
	else
	{
		sign_ = -1;
		num_ = (unsigned long)-i;
		denom_ = 1ul;
	}
}

Rational::Rational(long l)
{
	if (l >= 0l)
	{
		sign_ = 1;
		num_ = (unsigned long)l;
		denom_ = 1ul;
	}
	else
	{
		sign_ = -1;
		num_ = (unsigned long)-l;
		denom_ = 1ul;
	}
}

Rational::Rational(float f)
{  *this = Rational::toRational(double(f));  }

Rational::Rational(double d)
{  *this = Rational::toRational(d);  }

Rational::Rational(int num, int denom)
{  set(long(num), long(denom));  }

Rational::Rational(long num, long denom)
{  set(num, denom);  }

Rational::Rational(const Rational& rhs)
: sign_(rhs.sign_), num_(rhs.num_), denom_(rhs.denom_)
{}

Rational::~Rational()
{}

Rational& Rational::operator=(const Rational& rhs)
{
    if (this == &rhs) return *this;
    sign_ = rhs.sign_;
    num_ = rhs.num_;
    denom_ = rhs.denom_;
    return *this;
}

void Rational::set(int num, int denom)
{  set(long(num), long(denom));  }

void Rational::set(long num, long denom)
{
	if (denom == 0l)
	{
		throw LogException(LogException::L_ERROR, "not a number", __FILE__, __LINE__, __FUNCTION__);
	}
	
	sign_ = (num >= 0l && denom >= 0l) || (num <= 0l && denom <= 0l) ? 1 : -1;
	num_ = num >= 0l ? num : -num;
	denom_ = denom >= 0l ? denom : -denom;
	
	const unsigned long ulGCD = MathExt::gcd(num_, denom_);
	if (ulGCD != 1ul)
	{
		denom_ /= ulGCD;
		num_ /= ulGCD;
	}
}

void Rational::set(double num, double denom, double tolerance /*= MathConstant::EPS*/)
{
/*
	num = MathExt::round(num);
	denom = MathExt::round(denom);
*/
	if (-tolerance <= denom && denom <= tolerance)
	{
		throw LogException(LogException::L_ERROR, "not a number", __FILE__, __LINE__, __FUNCTION__);
	}
	if (Rational::isOverflow(fabs(num / denom)))
	{
		throw LogException(LogException::L_ERROR, "overflow", __FILE__, __LINE__, __FUNCTION__);
	}
/*
	sign_ = (num >= 0.0 && denom >= 0.0) || (num <= 0.0 && denom <= 0.0) ? 1 : -1;
	num_ = num >= 0.0 ? num : -num;
	denom_ = denom >= 0.0 ? denom : -denom;
	
	const unsigned long ulGCD = MathExt::gcd(num_, denom_);
	if (ulGCD != 1ul)
	{
		denom_ /= ulGCD;
		num_ /= ulGCD;
	}
*/
	*this = Rational::toRational(num / denom);
}

void Rational::set(const Rational& r)
{
	sign_ = r.sign_;
	num_ = r.num_;
	denom_ = r.denom_;
}

float Rational::toFloat() const
{
	if (denom_ == 0ul)
	{
		throw LogException(LogException::L_ERROR, "not a number", __FILE__, __LINE__, __FUNCTION__);
	}
	return float(num_) * float(sign_) / float(denom_);
}

double Rational::toDouble() const
{
	if (denom_ == 0ul)
	{
		throw LogException(LogException::L_ERROR, "not a number", __FILE__, __LINE__, __FUNCTION__);
	}
	return double(num_) * double(sign_) / double(denom_);
}

bool Rational::isZero(double tolerance /*= MathConstant::EPS*/) const
{
	if (num_ == 0ul) return true;
	else
		return MathUtil::isZero(toDouble(), tolerance);
}
bool Rational::isEqual(const Rational& r, double tolerance /*= MathConstant::EPS*/) const
{
	if (sign_ == r.sign_ && num_ == r.num_ && denom_ == r.denom_)
		return true;
	else
		return MathUtil::isZero(toDouble() - r.toDouble(), tolerance);
}

bool Rational::operator==(const Rational& rhs) const
//{  return num_*sign_*rhs.denom_ == denom_*rhs.num_*rhs.sign_;  }
{  return isEqual(rhs);  }

bool Rational::operator!=(const Rational& rhs) const
//{  return num_*sign_*rhs.denom_ != denom_*rhs.num_*rhs.sign_;  }
{  return !isEqual(rhs);  }

bool Rational::operator>(const Rational& rhs) const
//{  return num_*sign_*rhs.denom_ > denom_*rhs.num_*rhs.sign_;  }
{
	if (sign_ * rhs.sign_ < 0) return sign_ > rhs.sign_;
	else
	{
		const double dLhs = double(num_) * double(sign_) * double(rhs.denom_);
		const double dRhs = double(denom_) * double(rhs.num_) * double(rhs.sign_);
		return dLhs > dRhs;
	}
}

bool Rational::operator>=(const Rational& rhs) const
//{  return num_*sign_*rhs.denom_ >= denom_*rhs.num_*rhs.sign_;  }
{
	if (sign_ * rhs.sign_ < 0) return sign_ > rhs.sign_;
	else
	{
		const double dLhs = double(num_) * double(sign_) * double(rhs.denom_);
		const double dRhs = double(denom_) * double(rhs.num_) * double(rhs.sign_);
		return dLhs >= dRhs;
	}
}

bool Rational::operator<(const Rational& rhs) const
//{  return num_*sign_*rhs.denom_ < denom_*rhs.num_*rhs.sign_;  }
{  return !operator>=(rhs);  }

bool Rational::operator<=(const Rational& rhs) const
//{  return num_*sign_*rhs.denom_ <= denom_*rhs.num_*rhs.sign_;  }
{  return !operator>(rhs);  }

Rational& Rational::operator+()
{  return *this;  }

Rational Rational::operator+(const Rational& rhs) const
{
	Rational aRational;
	
	if (denom_ == rhs.denom_)
	{
		double dTmp = double(num_) * double(sign_) + double(rhs.num_) * double(rhs.sign_);
		aRational.sign_ = (int)MathUtil::sign(dTmp);
		dTmp *= double(aRational.sign_);
		if (Rational::isOverflow(dTmp))
		{
			throw LogException(LogException::L_ERROR, "overflow", __FILE__, __LINE__, __FUNCTION__);
			//return *this;
		}
		aRational.num_ = (unsigned long)dTmp;
		aRational.denom_ = denom_;
	}
	else
	{
		const unsigned long ulGCD = MathExt::gcd(denom_, rhs.denom_);
		double dTmp = (double(num_) * double(sign_) * double(rhs.denom_)
			   + double(rhs.num_) * double(rhs.sign_) * double(denom_)) / double(ulGCD);
		aRational.sign_ = (int)MathUtil::sign(dTmp);
		dTmp *= double(aRational.sign_);
		if (Rational::isOverflow(dTmp) || Rational::isOverflow(double(denom_) * double(rhs.denom_) / double(ulGCD)))
		{
			throw LogException(LogException::L_ERROR, "overflow", __FILE__, __LINE__, __FUNCTION__);
			//return *this;
		}
		aRational.num_ = (unsigned long)dTmp;
		// parenthesis is indispensable
		aRational.denom_ = denom_ * (rhs.denom_ / ulGCD);
	}
	
	aRational.makeMinimalRational();
	return aRational;
}

Rational& Rational::operator+=(const Rational& rhs)
{
	if (denom_ == rhs.denom_)
	{
		double dTmp = double(num_) * double(sign_) + double(rhs.num_) * double(rhs.sign_);
		sign_ = (int)MathUtil::sign(dTmp);
		dTmp *= double(sign_);
		if (Rational::isOverflow(dTmp))
		{
			throw LogException(LogException::L_ERROR, "overflow", __FILE__, __LINE__, __FUNCTION__);
			//return *this;
		}
		num_ = (unsigned long)dTmp;
		//denom_ = denom_;
	}
	else
	{
		const unsigned long ulGCD = MathExt::gcd(denom_, rhs.denom_);
		double dTmp = (double(num_) * double(sign_) * double(rhs.denom_)
			   + double(rhs.num_) * double(rhs.sign_) * double(denom_)) / double(ulGCD);
		sign_ = (int)MathUtil::sign(dTmp);
		dTmp *= double(sign_);
		if (Rational::isOverflow(dTmp) || Rational::isOverflow(double(denom_) * double(rhs.denom_) / double(ulGCD)))
		{
			throw LogException(LogException::L_ERROR, "overflow", __FILE__, __LINE__, __FUNCTION__);
			//return *this;
		}
		num_ = (unsigned long)dTmp;
		// parenthesis is indispensable
		denom_ *= (rhs.denom_ / ulGCD);
	}
	
	makeMinimalRational();
	return *this;
}

Rational Rational::operator-() const
{
	Rational aRational(*this);
	aRational.sign_ = -sign_;
	return aRational;
}

Rational Rational::operator-(const Rational& rhs) const
{
	Rational aRational;
	
	if (denom_ == rhs.denom_)
	{
		double dTmp = double(num_) * double(sign_) - double(rhs.num_) * double(rhs.sign_);
		aRational.sign_ = (int)MathUtil::sign(dTmp);
		dTmp *= double(aRational.sign_);
		if (Rational::isOverflow(dTmp))
		{
			throw LogException(LogException::L_ERROR, "overflow", __FILE__, __LINE__, __FUNCTION__);
			//return *this;
		}
		aRational.num_ = (unsigned long)dTmp;
		aRational.denom_ = denom_;
	}
	else
	{
		const unsigned long ulGCD = MathExt::gcd(denom_, rhs.denom_);
		double dTmp = (double(num_) * double(sign_) * double(rhs.denom_)
			   - double(rhs.num_) * double(rhs.sign_) * double(denom_)) / double(ulGCD);
		aRational.sign_ = (int)MathUtil::sign(dTmp);
		dTmp *= double(aRational.sign_);
		if (Rational::isOverflow(dTmp) || Rational::isOverflow(double(denom_) * double(rhs.denom_) / double(ulGCD)))
		{
			throw LogException(LogException::L_ERROR, "overflow", __FILE__, __LINE__, __FUNCTION__);
			//return *this;
		}
		aRational.num_ = (unsigned long)dTmp;
		// parenthesis is indispensable
		aRational.denom_ = denom_ * (rhs.denom_ / ulGCD);
	}
	
	aRational.makeMinimalRational();
	return aRational;
}

Rational& Rational::operator-=(const Rational& rhs)
{
	if (denom_ == rhs.denom_)
	{
		double dTmp = double(num_) * double(sign_) - double(rhs.num_) * double(rhs.sign_);
		sign_ = (int)MathUtil::sign(dTmp);
		dTmp *= double(sign_);
		if (Rational::isOverflow(dTmp))
		{
			throw LogException(LogException::L_ERROR, "overflow", __FILE__, __LINE__, __FUNCTION__);
			//return *this;
		}
		num_ = (unsigned long)dTmp;
		//denom_ = denom_;
	}
	else
	{
		const unsigned long ulGCD = MathExt::gcd(denom_, rhs.denom_);
		double dTmp = (double(num_) * double(sign_) * double(rhs.denom_)
			   - double(rhs.num_) * double(rhs.sign_) * double(denom_)) / double(ulGCD);
		sign_ = (int)MathUtil::sign(dTmp);
		dTmp *= double(sign_);
		if (Rational::isOverflow(dTmp) || Rational::isOverflow(double(denom_) * double(rhs.denom_) / double(ulGCD)))
		{
			throw LogException(LogException::L_ERROR, "overflow", __FILE__, __LINE__, __FUNCTION__);
			//return *this;
		}
		num_ = (unsigned long)dTmp;
		// parenthesis is indispensable
		denom_ *= (rhs.denom_ / ulGCD);
	}
	
	makeMinimalRational();
	return *this;
}

Rational Rational::operator*(const Rational& rhs) const
{
	if (num_ == 0ul || rhs.num_ == 0ul)	return Rational(0L);
	
	Rational aRational;
	
	unsigned long ulGCD = MathExt::gcd(num_, rhs.denom_);
	aRational.num_ = num_ / ulGCD;
	const unsigned long ulTmp1 = rhs.denom_ / ulGCD;
	
	ulGCD = MathExt::gcd(denom_, rhs.num_);
	aRational.denom_ = denom_ / ulGCD;
	const unsigned long ulTmp2 = rhs.num_ / ulGCD;
	
	if (Rational::isOverflow(double(aRational.num_) * double(ulTmp2)) || Rational::isOverflow(double(aRational.denom_) * double(ulTmp1)))
	{
		throw LogException(LogException::L_ERROR, "overflow", __FILE__, __LINE__, __FUNCTION__);
		//return *this;
	}

	aRational.sign_ = sign_ * rhs.sign_;
	aRational.num_ *= ulTmp2;
	aRational.denom_ *= ulTmp1;
	
	return aRational;
}

Rational& Rational::operator*=(const Rational& rhs)
{
	if (num_ == 0ul || rhs.num_ == 0ul)
	{
		set(0l, 1l);
		return *this;
	}
	
	unsigned long ulGCD = MathExt::gcd(num_, rhs.denom_);
	num_ /= ulGCD;
	const unsigned long ulTmp1 = rhs.denom_ / ulGCD;
	
	ulGCD = MathExt::gcd(denom_, rhs.num_);
	denom_ /= ulGCD;
	const unsigned long ulTmp2 = rhs.num_ / ulGCD;
	
	if (Rational::isOverflow(double(num_) * double(ulTmp2)) || Rational::isOverflow(double(denom_) * double(ulTmp1)))
	{
		throw LogException(LogException::L_ERROR, "overflow", __FILE__, __LINE__, __FUNCTION__);
		//return *this;
	}

	sign_ *= rhs.sign_;
	num_ *= ulTmp2;
	denom_ *= ulTmp1;
	
	return *this;
}

Rational Rational::operator/(const Rational& rhs) const
{
	if (rhs.num_ == 0ul)
	{
		throw LogException(LogException::L_ERROR, "divide by zero", __FILE__, __LINE__, __FUNCTION__);
		//return *this;
	}
	
	Rational aRational;
	
	unsigned long ulGCD = MathExt::gcd(num_, rhs.num_);
	aRational.num_ = num_ / ulGCD;
	const unsigned long ulTmp1 = rhs.num_ / ulGCD;
	
	ulGCD = MathExt::gcd(denom_, rhs.denom_);
	aRational.denom_ = denom_ / ulGCD;
	const unsigned long ulTmp2 = rhs.denom_ / ulGCD;
	
	if (Rational::isOverflow(double(aRational.num_) * double(ulTmp2)) || Rational::isOverflow(double(aRational.denom_) * double(ulTmp1)))
	{
		throw LogException(LogException::L_ERROR, "overflow", __FILE__, __LINE__, __FUNCTION__);
		//return *this;
	}

	aRational.sign_ = sign_ * rhs.sign_;
	aRational.num_ *= ulTmp2;
	aRational.denom_ *= ulTmp1;
	
	return aRational;
}

Rational& Rational::operator/=(const Rational& rhs)
{
	if (rhs.num_ == 0ul)
	{
		throw LogException(LogException::L_ERROR, "divide by zero", __FILE__, __LINE__, __FUNCTION__);
		//return *this;
	}
	
	unsigned long ulGCD = MathExt::gcd(num_, rhs.num_);
	num_ /= ulGCD;
	const unsigned long ulTmp1 = rhs.num_ / ulGCD;
	
	ulGCD = MathExt::gcd(denom_, rhs.denom_);
	denom_ /= ulGCD;
	const unsigned long ulTmp2 = rhs.denom_ / ulGCD;
	
	if (Rational::isOverflow(double(num_) * double(ulTmp2)) || Rational::isOverflow(double(denom_) * double(ulTmp1)))
	{
		throw LogException(LogException::L_ERROR, "overflow", __FILE__, __LINE__, __FUNCTION__);
		//return *this;
	}

	sign_ *= rhs.sign_;
	num_ *= ulTmp2;
	denom_ *= ulTmp1;
	
	return *this;
}
/*
Rational operator+(int i, Rational& rhs)
{  return Rational(i) + rhs;  }

Rational operator-(int i, Rational& rhs)
{  return Rational(i) - rhs;  }

Rational operator*(int i, Rational& rhs)
{  return Rational(i) * rhs;  }

Rational operator/(int i, Rational& rhs)
{  return Rational(integer) / rhs;  }

Rational operator+(long l, Rational& rhs)
{  return Rational(l) + rhs;  }

Rational operator-(long l, Rational& rhs)
{  return Rational(l) - rhs;  }

Rational operator*(long l, Rational& rhs)
{  return Rational(l) * rhs;  }

Rational operator/(long l, Rational& rhs)
{  return Rational(l) / rhs;  }

Rational operator+(float f, Rational& rhs)
{  return Rational(real) + rhs;  }

Rational operator-(float f, Rational& rhs)
{  return Rational(f) - rhs;  }

Rational operator*(float f, Rational& rhs)
{  return Rational(f) * rhs;  }

Rational operator/(float f, Rational& rhs)
{  return Rational(f) / rhs;  }

Rational operator+(double d, Rational& rhs)
{  return Rational(d) + rhs;  }

Rational operator-(double d, Rational& rhs)
{  return Rational(d) - rhs;  }

Rational operator*(double d, Rational& rhs)
{  return Rational(d) * rhs;  }

Rational operator/(double d, Rational& rhs)
{  return Rational(d) / rhs;  }
*/
void Rational::makeMinimalRational()
{
	const unsigned long ulGCD = MathExt::gcd(num_, denom_);
	if (ulGCD != 1ul)
	{
		num_ /= ulGCD;
		denom_ /= ulGCD;
	}
}

/*static*/ bool Rational::isOverflow(double value)
{  return fabs(value) > (double)std::numeric_limits<unsigned long>::max();  }

/*static*/ Rational Rational::toRational(double r, double tolerance /*= MathConstant::EPS*/)
// r(i-1) = d(i-1) - 1 / r(i), r(i) = 1 / (r(i-1) - d(i-1))  (i = 1, 2, ..., k)
// d(i) = round(r(i))
// r(0) = a
// r(k) = d(k)
{
/*
	double tmp = r;
	std::deque<long> d;

	int i = 0;
	do
	{
		d.push_back((long)MathExt::round(tmp));
		tmp = tmp - d[i];
		if (-tolerance <= tmp && tmp <= tolerance) break;
		tmp = 1.0 / tmp;
		++i;
	} while (i < MathConstant::ITERATION_LIMIT);

	std::deque<long>::size_type k = d.size();
	if (k == 1) return Rational(d[0], 1l);

	Rational f(d[k-1], 1l);
	double dVal;
	for (i = k-2 ; i >= 0 ; --i)
	{
		dVal = double(d[i]) * double(f.num()) + double(f.denom());
		if (Rational::isOverflow(dVal))
		{
			if (Rational::isOverflow(dVal / double(f.num())))
			{
				throw LogException(LogException::L_ERROR, "overflow", __FILE__, __LINE__, __FUNCTION__);
				//return f;
			}
			else f.set(dVal, double(f.num()));
		}
		else f.set(d[i] * f.num() + f.denom(), f.num());
	}
	return f;
*/
	double tmp = r, deviation;
	std::deque<long> d;
	Rational f;

	int i = 0;
	do
	{
		d.push_back((long)MathExt::round(tmp));
		tmp = tmp - d[i];
		Rational::calcContinuousRational(d, f);
		deviation = f.toDouble() - r;
		if (MathUtil::isZero(deviation, tolerance) || MathUtil::isZero(tmp, tolerance)) break;
		tmp = 1.0 / tmp;
		++i;
	} while (i < MathConstant::ITERATION_LIMIT);

	//Rational::calcContinuousRational(d, f);
	return f;
}

/*static*/ void Rational::calcContinuousRational(const std::deque<long>& denomCtr, Rational& r)
{
	const std::deque<long>::size_type nDenom = denomCtr.size();
	if (nDenom == 1)
	{
		r.set(denomCtr[0], 1l);
		return;
	}

	r.set(denomCtr[nDenom-1], 1l);
	for (int i = (int)nDenom-2 ; i >= 0 ; --i)
	{
		const double dVal = double(denomCtr[i]) * double(r.num()) + double(r.denom());
		if (Rational::isOverflow(dVal))
		{
			if (Rational::isOverflow(dVal / double(r.num())))
			{
				throw LogException(LogException::L_ERROR, "overflow", __FILE__, __LINE__, __FUNCTION__);
				//r.set(0l, 1l);
				//return;
			}
			else r.set(dVal, double(r.num()));
		}
		else r.set(denomCtr[i] * r.num() + r.denom(), r.num());
	}
}

//-----------------------------------------------------------------------------------------
// Rational Number API

std::istream& operator>>(std::istream& stream, Rational& r)
{
	// < n / d > means a rational number
	long lNum, lDen;
	char ch;
	stream >> ch >> lNum >> ch >> lDen >> ch;
	
	if (lDen == 0l)
	{
		throw LogException(LogException::L_ERROR, "not a number", __FILE__, __LINE__, __FUNCTION__);
		//r.set(0l, 1l);
		//return stream;
	}
	
	r.set(lNum, lDen);
	return stream;
}

std::ostream& operator<<(std::ostream& stream, const Rational& r)
{
	// < n / d > means a rational number
	stream << "< " << r.num() << " / " << r.denom() << " >";
	return stream;
}

Rational sqrt(const Rational& r, double tolerance /*= MathConstant::EPS*/)
{  return Rational::toRational(::sqrt(r.toDouble()), tolerance);  }

Rational pow(const Rational& r1, const Rational& r2, double tolerance /*= MathConstant::EPS*/)
// If the 1st argument passed to pow is real and less than 0
// and 2nd argment is not a whole number(integer), or you call pow(0, 0),
// a calculation of pow failed
{  return Rational::toRational(::pow(r1.toDouble(), r2.toDouble()), tolerance);  }

//Rational pow10(const Rational& r, double tolerance /*= MathConstant::EPS*/)
//{  return Rational::toRational(::pow(10.0, r.toDouble()), tolerance);  }

Rational exp(const Rational& r, double tolerance /*= MathConstant::EPS*/)
{  return Rational::toRational(::exp(r.toDouble()), tolerance);  }

Rational ln(const Rational& r, double tolerance /*= MathConstant::EPS*/)
{
	const double dVal = r.toDouble();
	if (dVal <= tolerance)
	{
		throw LogException(LogException::L_ERROR, "domain error", __FILE__, __LINE__, __FUNCTION__);
		//return r;
	}
	return Rational::toRational(::log(dVal), tolerance);
}

Rational log(const Rational& r, double tolerance /*= MathConstant::EPS*/)
{  return ln(r, tolerance);  }

Rational log10(const Rational& r, double tolerance /*= MathConstant::EPS*/)
{  return logb(10.0, r, tolerance);  }

Rational logb(double base, const Rational& r, double tolerance /*= MathConstant::EPS*/)
{
	if (base <= 0.0 || (-tolerance <= base-1.0 && base-1.0 <= tolerance))
	{
		throw LogException(LogException::L_ERROR, "illegal parameter value", __FILE__, __LINE__, __FUNCTION__);
		//return r;
	}
	const double dVal = r.toDouble();
	if (dVal <= tolerance)
	{
		throw LogException(LogException::L_ERROR, "domain error", __FILE__, __LINE__, __FUNCTION__);
		//return r;
	}
	return Rational::toRational(::log(dVal) / ::log(base), tolerance);
}

Rational sin(const Rational& r, double tolerance /*= MathConstant::EPS*/)
{  return Rational::toRational(::sin(r.toDouble()), tolerance);  }

Rational cos(const Rational& r, double tolerance /*= MathConstant::EPS*/)
{  return Rational::toRational(::cos(r.toDouble()), tolerance);  }

Rational tan(const Rational& r, double tolerance /*= MathConstant::EPS*/)
{  return Rational::toRational(::tan(r.toDouble()), tolerance);  }

Rational asin(const Rational& r, double tolerance /*= MathConstant::EPS*/)
{  return Rational::toRational(::asin(r.toDouble()), tolerance);  }

Rational acos(const Rational& r, double tolerance /*= MathConstant::EPS*/)
{  return Rational::toRational(::acos(r.toDouble()), tolerance);  }

Rational atan(const Rational& r, double tolerance /*= MathConstant::EPS*/)
{  return Rational::toRational(::atan(r.toDouble()), tolerance);  }

Rational atan2(const Rational& r1, const Rational& r2, double tolerance /*= MathConstant::EPS*/)
{  return Rational::toRational(::atan2(r1.toDouble(), r2.toDouble()), tolerance);  }

Rational sinh(const Rational& r, double tolerance /*= MathConstant::EPS*/)
{  return Rational::toRational(::sinh(r.toDouble()), tolerance);  }

Rational cosh(const Rational& r, double tolerance /*= MathConstant::EPS*/)
{  return Rational::toRational(::cosh(r.toDouble()), tolerance);  }

Rational tanh(const Rational& r, double tolerance /*= MathConstant::EPS*/)
{  return Rational::toRational(::tanh(r.toDouble()), tolerance);  }

Rational asinh(const Rational& r, double tolerance /*= MathConstant::EPS*/)
{  return Rational::toRational(MathExt::asinh(r.toDouble()), tolerance);  }

Rational acosh(const Rational& r, double tolerance /*= MathConstant::EPS*/)
{  return Rational::toRational(MathExt::acosh(r.toDouble()), tolerance);  }

Rational atanh(const Rational& r, double tolerance /*= MathConstant::EPS*/)
{  return Rational::toRational(MathExt::atanh(r.toDouble()), tolerance);  }

}  // namespace swl
