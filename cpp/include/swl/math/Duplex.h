#if !defined(__SWL_MATH__DUPLEX__H_)
#define __SWL_MATH__DUPLEX__H_ 1


#include "swl/math/MathExt.h"
#include "swl/math/MathUtil.h"
#include "swl/base/LogException.h"
#include <iostream>
#include <cmath>


namespace swl {

//-----------------------------------------------------------------------------------------
// class Duplex: dual numder & dual angle

template <typename T>
class Duplex
{
public:
    typedef T value_type;

public:
	Duplex(const T &tReal = T(0), const T &tDual = T(0))
	: real_(tReal), dual_(tDual)
	{}
	Duplex(const Duplex &rhs)
	: real_(rhs.real_), dual_(rhs.dual_)
	{}
	~Duplex()  {}
	
	Duplex & operator=(const Duplex &rhs)
	{
		if (this == &rhs) return *this;
		real_ = rhs.real_;
		dual_ = rhs.dual_;
		return *this;
	}

public:
	/// accessor & mutator
	T & real()  {  return real_;  }
	const T & real() const  {  return real_;  }
	T & dual()  {  return dual_;  }
	const T & dual() const  {  return dual_;  }
	
    ///
	bool isZero(const T &tTol = (T)MathConstant::EPS) const
	{  return MathUtil::isZero(real_, tTol) && MathUtil::isZero(dual_, tTol);  }
	bool isEqual(const Duplex &rhs, const T &tTol = (T)MathConstant::EPS) const
	{
		return MathUtil::isZero(real_ - rhs.real_, tTol) &&
			   MathUtil::isZero(dual_ - rhs.dual_, tTol);
	}
    
	/// comparison operator
	bool operator==(const Duplex &rhs)
    {  return isEqual(rhs);  }
    bool operator!=(const Duplex &rhs)
    {  return !isEqual(rhs);  }

	/// arithmetic operation
	Duplex & operator+()  {  return *this;  }
	Duplex operator+(const Duplex &rhs) const
	{  return Duplex(real_+rhs.real_, dual_+rhs.dual_);  }
	Duplex & operator+=(const Duplex &rhs)
	{
		real_ += rhs.real_;
		dual_ += rhs.dual_;
		return *this;
	}
	Duplex operator-() const
	{  return Duplex(-real_, -dual_);  }
	Duplex operator-(const Duplex &rhs) const
	{  return Duplex(real_-rhs.real_, dual_-rhs.dual_);  }
	Duplex & operator-=(const Duplex &rhs)
	{
		real_ -= rhs.real_;
		dual_ -= rhs.dual_;
		return *this;
	}
	Duplex operator*(const Duplex &rhs) const
	{  return Duplex<T>(real_*rhs.real_, real_*rhs.dual_ + dual_*rhs.real_);  }
	Duplex & operator*=(const Duplex &rhs)
	{  return *this = *this * rhs;  }
	Duplex operator/(const Duplex &rhs) const
	{
		if (MathUtil::isZero(rhs.real_))
		{
			throw LogException(LogException::L_ERROR, "divide by zero", __FILE__, __LINE__, __FUNCTION__);
			//return *this;
		}
		return Duplex<T>(real_ / rhs.real_, (dual_*rhs.real_ - real_*rhs.dual_) / (rhs.real_*rhs.real_));
	}
	Duplex & operator/=(const Duplex &rhs)
	{  return *this = *this / rhs;  }

	///
	T norm() const  {  return (T)std::sqrt(real_*real_ + dual_*dual_);  }
	Duplex conjugate() const  {  return Duplex(real_, -dual_);  }
	Duplex inverse() const
	{
		const T tValue(real_ * real_);

		if (MathUtil::isZero(tValue))
		{
			throw LogException(LogException::L_ERROR, "divide by zero", __FILE__, __LINE__, __FUNCTION__);
			//return *this;
		}

		return Duplex(T(1)/real_, -dual_/tValue);
		//return Duplex(real_/tValue, -dual_/tValue);
	}

private:
	/// the real part of a dual(duplex) number, real + {epsilon}*dual
	T real_;
	/// the dual part of a dual(duplex) number, real + {epsilon}*dual
	T dual_;
};


//-----------------------------------------------------------------------------------------
// Duplex Number API

template<typename T>
std::istream & operator>>(std::istream &stream, Duplex<T> &d)
{
	//  < r + d {e} > means d1.real() dual number
/*
	char ch, buf[16];
	stream >> ch >> d.real() >> ch >> d.dual() >> buf >> ch;
*/
	char ch;
	stream >> ch >> d.real() >> ch >> d.dual() >> ch >> ch >> ch >> ch;
	return stream;
}

template<typename T>
std::ostream & operator<<(std::ostream &stream, const Duplex<T> &d)
{
/*
	//  brace, <...> means d1.real() dual() number
	if (MathUtil::isZero(d.dual())) stream << "< " << d.real() << " >";
	else if (MathUtil::isZero(d.real())) stream << "< " << d.dual() << " {epsilon} >";
	else stream << "< " << d.real() << " + " << d.dual() << " {epsilon} >";
	return stream;
*/
	//  < r + d {e} > means d1.real() dual number
	stream << "< " << d.real() << " + " << d.dual() << " {e} >";
	return stream;
}

template<typename T>
Duplex<T> sqrt(const Duplex<T> &d)
{
	//if (MathUtil::isZero(d.real()))
	if (d.real() <= T(0))
	{
		throw LogException(LogException::L_ERROR, "divide by zero", __FILE__, __LINE__, __FUNCTION__);
		//return d;
	}

	const T tVal = (T)std::sqrt(d.real());
	return Duplex<T>(tVal, d.dual() / (T(2)*tVal));
}

template<typename T>
Duplex<T> pow(const Duplex<T> &d1, const Duplex<T> &d2)
{
/*
	if (d1.real() <= T(0))
	{
		throw LogException(LogException::L_ERROR, "invalid parameter value", __FILE__, __LINE__, __FUNCTION__);
		//return d1;
	}
	const T tVal = (T)std::pow(d1.real(), d2.real());
	return Duplex<T>(tVal, tVal * (d1.dual()*d2.real()/d1.real() + d2.dual()*(T)log(d1.real())));
*/
	return exp(d2 * ln(d1));
}
/*
template<typename T>
Duplex<T> pow10(const Duplex<T> &d)
{  return exp(d2 * ln(Duplex<T>(T(10))));  }
*/
template<typename T>
Duplex<T> exp(const Duplex<T> &d)
{
	const T tVal = (T)std::exp(d.real());
	return Duplex<T>(tVal, d.dual() * tVal);
}

template<typename T>
Duplex<T> ln(const Duplex<T> &d)
{
	if (d.real() <= T(0))
	{
		throw LogException(LogException::L_ERROR, "domain error", __FILE__, __LINE__, __FUNCTION__);
		//return d;
	}

	return Duplex<T>((T)std::log(d.real()), d.dual() / d.real());
}

template<typename T>
Duplex<T> log(const Duplex<T> &d)
{  return ln(d);  }

template<typename T>
Duplex<T> log10(const Duplex<T> &d)
{  return logb(T(10), d);  }

template<typename T>
Duplex<T> logb(const T &base, const Duplex<T> &d)
{
	if (base <= T(0) || MathUtil::isZero(base - T(1)))
	{
		throw LogException(LogException::L_ERROR, "invalid parameter value", __FILE__, __LINE__, __FUNCTION__);
		//return d;
	}
	if (d.real() <= T(0))
	{
		throw LogException(LogException::L_ERROR, "domain error", __FILE__, __LINE__, __FUNCTION__);
		//return d;
	}

	const T tVal(std::log(base));
	return Duplex<T>((T)std::log(d.real()) / tVal, d.dual() / (d.real() * tVal));
}

template<typename T>
Duplex<T> sin(const Duplex<T> &d)
{  return Duplex<T>((T)std::sin(d.real()), d.dual() * (T)std::cos(d.real()));  }

template<typename T>
Duplex<T> cos(const Duplex<T> &d)
{  return Duplex<T>((T)std::cos(d.real()), -d.dual() * (T)std::sin(d.real()));  }

template<typename T>
Duplex<T> tan(const Duplex<T> &d)
{
	const T tVal = (T)std::tan(d.real());
	return Duplex<T>(tVal, d.dual() * (T(1) + tVal*tVal));
}

template<typename T>
Duplex<T> asin(const Duplex<T> &d)
{
	if (d.real() <= -T(1) || d.real() >= T(1))
	{
		throw LogException(LogException::L_ERROR, "domain error", __FILE__, __LINE__, __FUNCTION__);
		//return d;
	}
	return Duplex<T>((T)std::asin(d.real()), d.dual() / (T)std::sqrt(T(1) - d.real()*d.real()));
}

template<typename T>
Duplex<T> acos(const Duplex<T> &d)
{
	if (d.real() <= -T(1) || d.real() >= T(1))
	{
		throw LogException(LogException::L_ERROR, "domain error", __FILE__, __LINE__, __FUNCTION__);
		//return d;
	}
	return Duplex<T>((T)std::acos(d.real()), -d.dual() / (T)std::sqrt(T(1) - d.real()*d.real()));
}

template<typename T>
Duplex<T> atan(const Duplex<T> &d)
{  return Duplex<T>((T)std::atan(d.real()), d.dual() / (T(1) + d.real()*d.real()));  }

template<typename T>
Duplex<T> sinh(const Duplex<T> &d)
{  return Duplex<T>((T)std::sinh(d.real()), d.dual() * (T)std::cosh(d.real()));  }

template<typename T>
Duplex<T> cosh(const Duplex<T> &d)
{  return Duplex<T>((T)std::cosh(d.real()), d.dual() * (T)std::sinh(d.real()));  }

template<typename T>
Duplex<T> tanh(const Duplex<T> &d)
{
	const T tVal = (T)std::tanh(d.real());
	return Duplex<T>(tVal, d.dual() * (T(1) - tVal*tVal));
}

template<typename T>
Duplex<T> asinh(const Duplex<T> &d)
{  return Duplex<T>((T)asinh(d.real()), d.dual() / (T)std::sqrt(d.real()*d.real() + T(1)));  }

template<typename T>
Duplex<T> acosh(const Duplex<T> &d)
// a domain of acosh, x >= 1
// when x < 1.0, a solution is a conmplex number
{
	//if (-T(1) <= d.real() && d.real() <= T(1))
	if (d.real() <= T(1))
	{
		throw LogException(LogException::L_ERROR, "domain error", __FILE__, __LINE__, __FUNCTION__);
		//return d;
	}
	return Duplex<T>((T)acosh(d.real()), d.dual() / (T)std::sqrt(d.real()*d.real() - T(1)));
}

template<typename T>
Duplex<T> atanh(const Duplex<T> &d)
// a domain of atanh, -1 < x < 1
// when x <= -1.0 || x >= 1.0, a solution is a conmplex number
{
/*
	const T tVal(T(1) - d.real()*d.real());
	if (MathUtil::isZero(tVal))
	{
		throw LogException(LogException::L_ERROR, "divide by zero", __FILE__, __LINE__, __FUNCTION__);
		//return d;
	}
	return Duplex<T>((T)atanh(d.real()), d.dual() / tVal);
*/
	if (d.real() <= -T(1) || d.real() >= T(1))
	{
		throw LogException(LogException::L_ERROR, "divide by zero", __FILE__, __LINE__, __FUNCTION__);
		//return d;
	}
	return Duplex<T>((T)atanh(d.real()), d.dual() / (T(1) - d.real()*d.real()));
}

}  // namespace swl


#endif  // __SWL_MATH__DUPLEX__H_
