#if !defined(__SWL_MATH__COMPLEX__H_)
#define __SWL_MATH__COMPLEX__H_ 1


#include "swl/math/MathExt.h"
#include "swl/base/LogException.h"
#include <iostream>
#include <cmath>


namespace swl {

//-----------------------------------------------------------------------------------------
// class Complex
		
template<typename T>
class Complex
{
public:
    typedef T value_type;
    
public:
	Complex(T tReal = T(0), T tImag = T(0))
	: real_(tReal), imag_(tImag)
	{}
	Complex(const Complex& rhs)
	: real_(rhs.real_), imag_(rhs.imag_)
	{}
	~Complex() {}
	
	Complex& operator=(const Complex& rhs)
	{
		if (this == &rhs) return *this;
		real_ = rhs.real_;
		imag_ = rhs.imag_;
		return *this;
	}

public:
	/// accessor & mutator
	T& real()  {  return real_;  }
	const T& real() const  {  return real_;  }
	T& imag()  {  return imag_;  }
	const T& imag() const  {  return imag_;  }
	
	///
	bool isZero(const T& tTol = (T)MathConstant::EPS) const
	{  return MathUtil::isZero(real_, tTol) && MathUtil::isZero(imag_, tTol);  }
	bool isEqual(const Complex& rhs, const T& tTol = (T)MathConstant::EPS) const
	{
		return MathUtil::isZero(real_ - rhs.real_, tTol) &&
			   MathUtil::isZero(imag_ - rhs.imag_, tTol);
	}
	
	/// comparison operator
	bool operator==(const Complex& rhs) const
	{  return isEqual(rhs);  }
	bool operator!=(const Complex& rhs) const
	{  return !isEqual(rhs);  }
	
	/// arithmetic operation
	Complex& operator+()  {  return *this;  }
	Complex operator+(const Complex& rhs) const
	{  return Complex(real_+rhs.real_, imag_+rhs.imag_);  }
	Complex& operator+=(const Complex& rhs)
	{
		real_ += rhs.real_;
		imag_ += rhs.imag_;
		return *this;
	}
	Complex operator-() const
	{  return Complex(-real_, -imag_);  }
	Complex operator-(const Complex& rhs) const
	{  return Complex(real_-rhs.real_, imag_-rhs.imag_);  }
	Complex& operator-=(const Complex& rhs)
	{
		real_ -= rhs.real_;
		imag_ -= rhs.imag_;
		return *this;
	}
	Complex operator*(const Complex& rhs) const
	{  return Complex(real_*rhs.real_-imag_*rhs.imag_, real_*rhs.imag_+imag_*rhs.real_);  }
	Complex& operator*=(const Complex& rhs)
	{  return *this = *this * rhs;  }
	Complex operator/(const Complex& rhs) const
	{
		if (MathUtil::isZero(rhs.real_) && MathUtil::isZero(rhs.imag_))
		{
			throw LogException(LogException::L_ERROR, "divide by zero", __FILE__, __LINE__, __FUNCTION__);
			//return *this;
		}
		
		const T tDenom = rhs.real_*rhs.real_ + rhs.imag_*rhs.imag_;
		return Complex<T>((real_*rhs.real_ + imag_*rhs.imag_) / tDenom, (imag_*rhs.real_ - real_*rhs.imag_) / tDenom);
	}
	Complex& operator/=(const Complex& rhs)
	{  return *this = *this / rhs;  }

	///
	T norm() const  {  return (T)::sqrt(real_*real_ + imag_*imag_);  }
	/// -pi <= angle <= pi
	T arg() const  {  return (T)std::atan2(imag_, real_);  }
	T mag() const  {  return norm();  }
	T abs() const  {  return norm();  }
	T amplitude() const  {  return norm();  }
	T angle() const  {  return arg();  }
	
	///
	Complex conjugate() const
	{  return Complex(real_, -imag_);  }
	Complex inverse() const
	{
		const T tValue(real_*real_ + imag_*imag_);

		if (MathUtil::isZero(tValue))
		{
			throw LogException(LogException::L_ERROR, "divide by zero", __FILE__, __LINE__, __FUNCTION__);
			//return *this;
		}

		return Complex(real_/tValue, -imag_/tValue);
	}

private:
	/// the real and imaginary parts of a complex number, real + {i}*imag
	T real_, imag_;
};


//-----------------------------------------------------------------------------------------
// Complex Number API
		
template<typename T>
std::istream& operator>>(std::istream& stream, Complex<T>& z)
{
	// < a + b {i} > means a complex number
/*
	char ch, buf[16];
	stream >> ch >> z.real() >> ch >> z.imag() >> buf >> ch;
*/
	char ch;
	stream >> ch >> z.real() >> ch >> z.imag() >> ch >> ch >> ch >> ch;
	return stream;
}

template<typename T>
std::ostream& operator<<(std::ostream& stream, const Complex<T>& z)
{
/*
	// brace, {...} means a complex number
	if (MathUtil::isZero(z.imag())) stream << "< " << z.real() << " >";
	else if (MathUtil::isZero(z.real())) stream << "< " << z.imag() << " {i} >";
	else stream << "< " << z.real() << " + " << z.imag() << " {i} >";
	return stream;
*/
	// < a + b {i} > means a complex number
	stream << "< " << z.real() << " + " << z.imag() << " {i} >";
	return stream;
}

template<typename T>
Complex<T> sqrt(const Complex<T>& z, int iIndex = 0)
// z^(1/n) = r^(1/n) * (cos((thet+2*k*pi)/n) + i*sin((thet+2*k*pi)/n))  (k = 0, 1, ... , n-1)
// when taking the principal value of arg(z) and k=0, the value of z^(1/n) is the principal value
{
	T tR((T)::sqrt(z.norm()));
	T tVal((z.arg() + T(MathConstant::_2_PI*iIndex)) / T(2));
	return Complex<T>(tR*(T)::cos(tVal), tR*(T)::sin(tVal));
}

template<typename T>
Complex<T> pow(const Complex<T>& z1, const Complex<T>& z2, int iIndex = 0)
// z1^z2 = exp(z2 * ln(z1))
{  return exp(z2 * ln(z1, iIndex));  }
/*
template<typename T>
Complex<T> pow10(const Complex<T>& z, int iIndex = 0)
// 10^z = exp(z * ln(10))
{  return exp(z * ln(Complex<T>(T(10)), iIndex));  }
*/
template<typename T>
Complex<T> exp(const Complex<T>& z)
// exp(a+b*i) = exp(a)*(cos(b) + i*sin(b))
{  return Complex<T>((T)::exp(z.real())*(T)::cos(z.imag()), (T)::exp(z.real())*(T)::sin(z.imag()));  }

template<typename T>
Complex<T> ln(const Complex<T>& z, int iIndex = 0)
// obtain k-th non-principal value
// ln(z) = Ln(z) + j * 2 * k * pi (k = ..., -2, -1, 0, 1, 2, ...)
// where Ln(z) : the principal value of ln(z), = ln(abs(z)) + j * arg(z)
{
	if (MathUtil::isZero(z.real()) && MathUtil::isZero(z.imag()))
	{
		throw LogException(LogException::L_ERROR, "domain error", __FILE__, __LINE__, __FUNCTION__);
		//return z;
	}
	return Complex<T>((T)::log(z.norm()), z.arg()+T(MathConstant::_2_PI*iIndex));
}

template<typename T>
Complex<T> log(const Complex<T>& z, int iIndex = 0)
{  return ln(z, iIndex);  }

template<typename T>
Complex<T> log10(const Complex<T>& z, int iIndex = 0)
{  return logb(T(10), z, iIndex);  }

template<typename T>
Complex<T> logb(const T& base, const Complex<T>& z, int iIndex = 0)
// obtain k-th non-principal value
// ln(z) = Ln(z) + j * 2 * k * pi (k = ..., -2, -1, 0, 1, 2, ...)
// where Ln(z) : the principal value of ln(z), = ln(abs(z)) + j * arg(z)
{
	if (base <= T(0))
	{
		throw LogException(LogException::L_ERROR, "illegal parameter value", __FILE__, __LINE__, __FUNCTION__);
		//return z;
	}
	if (MathUtil::isZero(z.real()) && MathUtil::isZero(z.imag()))
	{
		throw LogException(LogException::L_ERROR, "domain error", __FILE__, __LINE__, __FUNCTION__);
		//return z;
	}
	const T tVal(::log(base));
	return Complex<T>((T)::log(z.norm())/tVal, (z.arg()+T(MathConstant::_2_PI*iIndex))/tVal);
}

template<typename T>
Complex<T> sin(const Complex<T>& z)
// sin(-z) = -sin(z)
{  return Complex<T>((T)::sin(z.real())*(T)::cosh(z.imag()), (T)::cos(z.real())*(T)::sinh(z.imag()));  }

template<typename T>
Complex<T> cos(const Complex<T>& z)
// cos(-z) = cos(z)
{  return Complex<T>((T)::cos(z.real())*(T)::cosh(z.imag()), -(T)::sin(z.real())*(T)::sinh(z.imag()));  }

template<typename T>
Complex<T> tan(const Complex<T>& z)
{  return sin(z) / cos(z);  }

template<typename T>
Complex<T> asin(const Complex<T>& z, int iIndex = 0)
// the principal value(w0) of w = u + v*i = asin(z) is defined to be the value
// for which -pi/2 <= u <= pi/2 when v >= 0 and -pi/2 < u < pi/2 when v < 0
// w = w0 + 2 * k * pi or w = (1 + 2 * k) * pi - w0 (k = ..., -2, -1, 0, 1, 2, ...)
{
	return -Complex<T>(T(0),T(1)) * ln(Complex<T>(T(0),T(1))*z + sqrt(Complex<T>(T(1)) - z*z, iIndex), iIndex);
	//return -Complex<T>(T(0),T(1)) * ln(Complex<T>(T(0),T(1))*z - sqrt(Complex<T>(T(1)) - z*z, iIndex), iIndex);
}

template<typename T>
Complex<T> acos(const Complex<T>& z, int iIndex = 0)
{
	return -Complex<T>(T(0),T(1)) * ln(z + sqrt(z*z - Complex<T>(T(1)), iIndex), iIndex);
	//return -Complex<T>(T(0),T(1)) * ln(z - sqrt(z*z - Complex<T>(T(1)), iIndex), iIndex);
}

template<typename T>
Complex<T> atan(const Complex<T>& z)
{
	return (Complex<T>(T(0),T(1)) / Complex<T>(T(2))) * ln((Complex<T>(T(0),T(1))+z) / (Complex<T>(T(0),T(1))-z));
}

template<typename T>
Complex<T> sinh(const Complex<T>& z)
// sinh(i*z) = i*sin(z), sin(i*z) = i*sinh(z)
// sinh(-z) = -sinh(z)
{  return Complex<T>((T)::sinh(z.real())*(T)::cos(z.imag()), (T)::cosh(z.real())*(T)::sin(z.imag()));  }

template<typename T>
Complex<T> cosh(const Complex<T>& z)
// cosh(i*z) = cos(z), cos(i*z) = cosh(z)
// cosh(-z) = cosh(z)
{  return Complex<T>((T)::cosh(z.real())*(T)::cos(z.imag()), (T)::sinh(z.real())*(T)::sin(z.imag()));  }

template<typename T>
Complex<T> tanh(const Complex<T>& z)
{  return sinh(z) / cosh(z);  }

template<typename T>
Complex<T> asinh(const Complex<T>& z, int iIndex = 0)
{
	return ln(z + sqrt(z*z + Complex<T>(T(1)), iIndex), iIndex);
	//return ln(z - sqrt(z*z + Complex<T>(T(1)), iIndex), iIndex);
}

template<typename T>
Complex<T> acosh(const Complex<T>& z, int iIndex = 0)
{
	return ln(z + sqrt(z*z - Complex<T>(T(1)), iIndex), iIndex);
	//return ln(z - sqrt(z*z - Complex<T>(T(1)), iIndex), iIndex);
}

template<typename T>
Complex<T> atanh(const Complex<T>& z)
{  return ln((Complex<T>(T(1)) + z) / (Complex<T>(T(1)) - z)) / Complex<T>(T(2));  }

}  // namespace swl


#endif  // __SWL_MATH__COMPLEX__H_
