#include "swl/math/MathExt.h"
#include "swl/math/MathUtil.h"
#include <sstream>
#include <limits>
#include <cmath>


#if defined(WIN32) && defined(_DEBUG)
void * __cdecl operator new(size_t nSize, const char *lpszFileName, int nLine);
#define new new(__FILE__, __LINE__)
//#pragma comment(lib, "mfc80ud.lib")
#endif

#if defined(max)
#	undef max
#endif


namespace swl {

//-----------------------------------------------------------------------------------------
//

double round(const double x)
// round(-2.51) = -3, round(-2.50) = -2, round(-2.49) = -2
// round(2.49) = 2, round(2.50) = 3, round(2.51) = 3
{
	return std::floor(x + 0.5);
}

double round(const double x, const int decimalPlace)
// round(x, 0) = round(x)
// round(12340.98, 3) is 12000.00
// round(0.003456, -4) is 0.003500
{
	return round(x * std::pow(0.1, decimalPlace)) * std::pow(10.0, decimalPlace);
}

double logb(const double base, const double x)
{
	const double dTol = MathConstant::EPS;
	if (base <= 0.0 || (-dTol <= base - 1.0 && base - 1.0 <= dTol))
	{
		std::ostringstream stream;
		stream << "swl::logb() at " << __LINE__ << " in " << __FILE__;
		throw std::invalid_argument(stream.str().c_str());
	}
	if (x <= 0.0)
	{
		std::ostringstream stream;
		stream << "swl::logb() at " << __LINE__ << " in " << __FILE__;
		throw std::domain_error(stream.str().c_str());
	}

	return std::log(x) / std::log(base);
}

double asinh(const double x)
{  return std::log(x + std::sqrt(x * x + 1.0));  }

double acosh(const double x)
{
	if (x < 1.0)
	{
		// when x < 1.0, a solution is a conmplex number
		std::ostringstream stream;
		stream << "swl::acosh() at " << __LINE__ << " in " << __FILE__;
		throw std::domain_error(stream.str().c_str());
	}

	return std::log(x + std::sqrt(x * x - 1.0));
	//return std::log(x - std::sqrt(x * x - 1.0));
}

double atanh(const double x)
{
	if (x <= -1.0 || x >= 1.0)
	{
		// when x <= -1.0 || x >= 1.0, a solution is a conmplex number
		std::ostringstream stream;
		stream << "swl::atanh() at " << __LINE__ << " in " << __FILE__;
		throw std::domain_error(stream.str().c_str());
	}

	return std::log(std::sqrt((1.0 + x) / (1.0 - x)));
}


//-----------------------------------------------------------------------------------------
// struct MathExt

double MathExt::mantissa(const double x, double *pow /*= NULL*/)
// value = mentissa * 10^pow
{
	const double dSign = MathUtil::sign(x);
	double x2 = std::fabs(x2);

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

unsigned long MathExt::gcd(const unsigned long lhs, const unsigned long rhs)
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

double MathExt::gcd(const double lhs, const double rhs, const double dTol /*= MathConstant::EPS*/)
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

unsigned long MathExt::lcm(const unsigned long lhs, const unsigned long rhs)
// LCM: least common multiplier
{  return lhs * rhs / gcd(lhs, rhs);  }

double MathExt::lcm(const double lhs, const double rhs, const double dTol /*= MathConstant::EPS*/)
// LCM: least common multiplier
{  return lhs * rhs / gcd(lhs, rhs, dTol);  }

unsigned long MathExt::factorial(const unsigned long n)
{
	unsigned long ulFactorial = 1ul;
	
	if (n == 0ul || n == 1ul) return 1ul;
	for (unsigned long k = n; k > 1ul; --k)
	{
		if (std::numeric_limits<unsigned long>::max() / ulFactorial < k)
		{
			std::ostringstream stream;
			stream << "swl::MathExt::factorial() at " << __LINE__ << " in " << __FILE__;
			throw std::overflow_error(stream.str().c_str());
		}
		ulFactorial *= k;
	}
	return ulFactorial;
}

double MathExt::factorial(const double n, const double dTol /*= MathConstant::EPS*/)
{
	double dFactorial = 1.0;
	
	if ((-dTol <= n && n <= dTol) || (-dTol <= n-1.0 && n-1.0 <= dTol)) return 1.0;
	for (unsigned long k = (unsigned long)round(n); k > 1ul; --k)
	{
		if (std::numeric_limits<double>::max() / dFactorial < k)
		{
			std::ostringstream stream;
			stream << "swl::MathExt::factorial() at " << __LINE__ << " in " << __FILE__;
			throw std::overflow_error(stream.str().c_str());
		}
		dFactorial *= k;
	}
	return dFactorial;
}

unsigned long MathExt::permutation(const unsigned long lhs, const unsigned long rhs)
{
	if (lhs < rhs)
	{
		std::ostringstream stream;
		stream << "swl::MathExt::permutation() at " << __LINE__ << " in " << __FILE__;
		throw std::invalid_argument(stream.str().c_str());
	}
	return factorial(lhs) / factorial(lhs - rhs); 
}

double MathExt::permutation(const double lhs, const double rhs,const  double dTol /*= MathConstant::EPS*/)
{
	if (lhs < rhs)
	{
		std::ostringstream stream;
		stream << "swl::MathExt::permutation() at " << __LINE__ << " in " << __FILE__;
		throw std::invalid_argument(stream.str().c_str());
	}
	return factorial(lhs, dTol) / factorial(lhs - rhs, dTol); 
}

unsigned long MathExt::binomial(const unsigned long lhs, const unsigned long rhs)
{
	if (lhs < rhs)
	{
		std::ostringstream stream;
		stream << "swl::MathExt::binomial() at " << __LINE__ << " in " << __FILE__;
		throw std::invalid_argument(stream.str().c_str());
	}
	return factorial(lhs) / (factorial(lhs - rhs) * factorial(rhs));
}

//  combination or binomial
double MathExt::binomial(const double lhs, const double rhs, const double dTol /*= MathConstant::EPS*/)
{
	if (lhs < rhs)
	{
		std::ostringstream stream;
		stream << "swl::MathExt::binomial() at " << __LINE__ << " in " << __FILE__;
		throw std::invalid_argument(stream.str().c_str());
	}
	return factorial(lhs, dTol) / (factorial(lhs - rhs, dTol) * factorial(rhs, dTol));
}

}  // namespace swl
