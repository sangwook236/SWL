#include "swl/Config.h"
#include "swl/math/MathExt.h"
#include "swl/math/MathUtil.h"
#include "swl/base/LogException.h"
//#include <sstream>
#include <limits>
#include <cmath>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif

#if defined(max)
#	undef max
#endif


namespace swl {

//-----------------------------------------------------------------------------------------
// struct MathExt

/*static*/ double MathExt::mantissa(const double x, double *pow /*= NULL*/)
// value = mentissa * 10^pow
{
	const double dSign = MathUtil::sign(x);
	double x2 = std::fabs(x);

	if (x2 >= 1.0)
		// 123.45 = 1.2345 * 10^2
		for (unsigned long i = 0ul; i <= std::numeric_limits<unsigned long>::max(); ++i)
		{
			if (x2 < 10.0)
			{
				if (pow) *pow = double(i);
				return x2 * dSign;
			}
			x2 /= 10.0;
		}
	else
		// 0.006789 = 6.789 * 10^-3
		for (unsigned long i = 1ul; i <= std::numeric_limits<unsigned long>::max(); ++i)
		{
			x2 *= 10.0;
			if (x2 >= 1.0)
			{
				if (pow) *pow = -double(i);
				return x2 * dSign;
			}
		}
			
	if (pow) *pow = 0.0;
	return 0.0;
}

/*static*/ double MathExt::round(const double x)
// round(-2.51) = -3, round(-2.50) = -2, round(-2.49) = -2
// round(2.49) = 2, round(2.50) = 3, round(2.51) = 3
{
	return std::floor(x + 0.5);
}

/*static*/ double MathExt::round(const double x, const int decimalPlace)
// round(x, 0) = round(x)
// round(12340.98, 3) is 12000.00
// round(0.003456, -4) is 0.003500
{
	return round(x * std::pow(0.1, decimalPlace)) * std::pow(10.0, decimalPlace);
}

/*static*/ double MathExt::logb(const double base, const double x)
{
	const double dTol = MathConstant::EPS;
	if (base <= 0.0 || (-dTol <= base - 1.0 && base - 1.0 <= dTol))
	{
		//std::ostringstream stream;
		//stream << "swl::MathExt::logb() at " << __LINE__ << " in " << __FILE__;
		//throw std::invalid_argument(stream.str().c_str());
		throw LogException(LogException::L_ERROR, "invalid argument", __FILE__, __LINE__, __FUNCTION__);
	}
	if (x <= 0.0)
	{
		//std::ostringstream stream;
		//stream << "swl::MathExt::logb() at " << __LINE__ << " in " << __FILE__;
		//throw std::domain_error(stream.str().c_str());
		throw LogException(LogException::L_ERROR, "domain error", __FILE__, __LINE__, __FUNCTION__);
	}

	return std::log(x) / std::log(base);
}

/*static*/ double MathExt::asinh(const double x)
{  return std::log(x + std::sqrt(x * x + 1.0));  }

/*static*/ double MathExt::acosh(const double x)
{
	if (x < 1.0)
	{
		// when x < 1.0, a solution is a conmplex number
		//std::ostringstream stream;
		//stream << "swl::MathExt::acosh() at " << __LINE__ << " in " << __FILE__;
		//throw std::domain_error(stream.str().c_str());
		throw LogException(LogException::L_ERROR, "domain error", __FILE__, __LINE__, __FUNCTION__);
	}

	return std::log(x + std::sqrt(x * x - 1.0));
	//return std::log(x - std::sqrt(x * x - 1.0));
}

/*static*/ double MathExt::atanh(const double x)
{
	if (x <= -1.0 || x >= 1.0)
	{
		// when x <= -1.0 || x >= 1.0, a solution is a conmplex number
		//std::ostringstream stream;
		//stream << "swl::MathExt::atanh() at " << __LINE__ << " in " << __FILE__;
		//throw std::domain_error(stream.str().c_str());
		throw LogException(LogException::L_ERROR, "domain error", __FILE__, __LINE__, __FUNCTION__);
	}

	return std::log(std::sqrt((1.0 + x) / (1.0 - x)));
}

/*static*/ unsigned long MathExt::gcd(const unsigned long lhs, const unsigned long rhs)
// GCD: greatest common divisor
// use Euclidean algorithm
// if return value is one(1), GCD does not exist
{
	unsigned long lhs2 = lhs;
	unsigned long rhs2 = rhs;
	while (true)
	{
		if (!(lhs2 %= rhs2)) return rhs2;
		if (!(rhs2 %= lhs2)) return lhs2;
	} 
}

/*static*/ double MathExt::gcd(const double lhs, const double rhs, const double dTol /*= MathConstant::EPS*/)
// GCD: greatest common divisor
// use Euclidean algorithm
// if return value is one(1), GCD does not exist
{
	double lhs2 = round(lhs);
	double rhs2 = round(rhs);

	while (true)
	{
		//lhs2 = std::fmod(lhs2, rhs2);
		lhs2 = round(std::fmod(lhs2, rhs2));
		if (-dTol <= lhs2 && lhs2 <= dTol) return rhs2;
		//rhs2 = std::fmod(rhs2, lhs2);
		rhs2 = round(std::fmod(rhs2, lhs2));
		if (-dTol <= rhs2 && rhs2 <= dTol) return lhs2;
	}
}

/*static*/ unsigned long MathExt::lcm(const unsigned long lhs, const unsigned long rhs)
// LCM: least common multiplier
{  return lhs * rhs / gcd(lhs, rhs);  }

/*static*/ double MathExt::lcm(const double lhs, const double rhs, const double dTol /*= MathConstant::EPS*/)
// LCM: least common multiplier
{  return lhs * rhs / gcd(lhs, rhs, dTol);  }

/*static*/ unsigned long MathExt::factorial(const unsigned long n)
{
	unsigned long ulFactorial = 1ul;
	
	if (n == 0ul || n == 1ul) return 1ul;
	for (unsigned long k = n; k > 1ul; --k)
	{
		if (std::numeric_limits<unsigned long>::max() / ulFactorial < k)
		{
			//std::ostringstream stream;
			//stream << "swl::MathExt::factorial() at " << __LINE__ << " in " << __FILE__;
			//throw std::overflow_error(stream.str().c_str());
			throw LogException(LogException::L_ERROR, "overflow error", __FILE__, __LINE__, __FUNCTION__);
		}
		ulFactorial *= k;
	}
	return ulFactorial;
}

/*static*/ double MathExt::factorial(const double n, const double dTol /*= MathConstant::EPS*/)
{
	double dFactorial = 1.0;
	
	if ((-dTol <= n && n <= dTol) || (-dTol <= n - 1.0 && n - 1.0 <= dTol)) return 1.0;
	for (unsigned long k = (unsigned long)round(n); k > 1ul; --k)
	{
		if (std::numeric_limits<double>::max() / dFactorial < k)
		{
			//std::ostringstream stream;
			//stream << "swl::MathExt::factorial() at " << __LINE__ << " in " << __FILE__;
			//throw std::overflow_error(stream.str().c_str());
			throw LogException(LogException::L_ERROR, "overflow error", __FILE__, __LINE__, __FUNCTION__);
		}
		dFactorial *= k;
	}
	return dFactorial;
}

/*static*/ unsigned long MathExt::permutation(const unsigned long lhs, const unsigned long rhs)
{
	if (lhs < rhs)
	{
		//std::ostringstream stream;
		//stream << "swl::MathExt::permutation() at " << __LINE__ << " in " << __FILE__;
		//throw std::invalid_argument(stream.str().c_str());
		throw LogException(LogException::L_ERROR, "invalid argument", __FILE__, __LINE__, __FUNCTION__);
	}
	return factorial(lhs) / factorial(lhs - rhs); 
}

/*static*/ double MathExt::permutation(const double lhs, const double rhs, const double dTol /*= MathConstant::EPS*/)
{
	if (lhs < rhs)
	{
		//std::ostringstream stream;
		//stream << "swl::MathExt::permutation() at " << __LINE__ << " in " << __FILE__;
		//throw std::invalid_argument(stream.str().c_str());
		throw LogException(LogException::L_ERROR, "invalid argument", __FILE__, __LINE__, __FUNCTION__);
	}
	return factorial(lhs, dTol) / factorial(lhs - rhs, dTol); 
}

/*static*/ unsigned long MathExt::binomial(const unsigned long lhs, const unsigned long rhs)
{
	if (lhs < rhs)
	{
		//std::ostringstream stream;
		//stream << "swl::MathExt::binomial() at " << __LINE__ << " in " << __FILE__;
		//throw std::invalid_argument(stream.str().c_str());
		throw LogException(LogException::L_ERROR, "invalid argument", __FILE__, __LINE__, __FUNCTION__);
	}
	return factorial(lhs) / (factorial(lhs - rhs) * factorial(rhs));
}

// combination or binomial
/*static*/ double MathExt::binomial(const double lhs, const double rhs, const double dTol /*= MathConstant::EPS*/)
{
	if (lhs < rhs)
	{
		//std::ostringstream stream;
		//stream << "swl::MathExt::binomial() at " << __LINE__ << " in " << __FILE__;
		//throw std::invalid_argument(stream.str().c_str());
		throw LogException(LogException::L_ERROR, "invalid argument", __FILE__, __LINE__, __FUNCTION__);
	}
	return factorial(lhs, dTol) / (factorial(lhs - rhs, dTol) * factorial(rhs, dTol));
}

/*static*/ double MathExt::det(const double a, const double b, const double c, const double d)
{
	return a * d - b * c;
}

/*static*/ double MathExt::det(const double a, const double b, const double c, const double d, const double e, const double f, const double g, const double h, const double i)
{
	return a * e * i - a * f * h - b * d * i + b * f * g + c * d * h - c * e * g;
}


/*static*/ double MathExt::centralAngle(const double longitude1, const double latitude1, const double longitude2, const double latitude2, const double tol /*= MathConstant::EPS*/)
{
    const double sin_phi1 = std::sin(latitude1), cos_phi1 = std::cos(latitude1);
    const double sin_phi2 = std::sin(latitude2), cos_phi2 = std::cos(latitude2);
#if 0
    const double delta_lambda = std::abs(longitude2 - longitude1);
#else
    double delta_lambda = std::abs(MathUtil::wrap(longitude2, 0.0, 2.0*MathConstant::PI, tol) - MathUtil::wrap(longitude1, 0.0, 2.0*MathConstant::PI, tol));
    if (delta_lambda > MathConstant::PI) delta_lambda = 2.0*MathConstant::PI - delta_lambda;
#endif
    const double sin_delta_lambda = std::sin(delta_lambda), cos_delta_lambda = std::cos(delta_lambda);

    const double num1 = cos_phi2 * sin_delta_lambda, num2 = cos_phi1 * sin_phi2 - sin_phi1 * cos_phi2 * cos_delta_lambda;
    // TODO [check] >>
    const double num = std::sqrt(num1*num1 + num2*num2);
    //const double num = -std::sqrt(num1*num1 + num2*num2);
    const double den = sin_phi1 * sin_phi2 + cos_phi1 * cos_phi2 * cos_delta_lambda;

    return std::atan2(num, den);
}

}  // namespace swl
