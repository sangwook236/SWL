#include "swl/math/MathUtil.h"
#include <cmath>
#include <algorithm>


#if defined(WIN32) && defined(_DEBUG)
void * __cdecl operator new(size_t nSize, const char* lpszFileName, int nLine);
#define new new(__FILE__, __LINE__)
//#pragma comment(lib, "mfc80ud.lib")
#endif

#if defined(PI)  // for djgpp
#	undef PI
#endif


namespace swl {

namespace {

static std::string convert_base_field(const unsigned long dec, const unsigned long base)
{
	std::string num;
	unsigned long dec2 = dec;
	unsigned long remainder;
	while (dec2 >= base)
	{
		remainder = dec2 % base;
		num += std::string::value_type(remainder >= 10ul ? 'A' + remainder - 10ul : '0' + remainder);
		dec2 /= base;
	}
	num += std::string::value_type(dec2 >= 10ul ? 'A' + dec2 - 10ul : '0' + dec2);
	std::reverse(num.begin(), num.end());
	return num;
}

}  // unnamed namespace

//-----------------------------------------------------------------------------------------
// struct MathUtil

bool MathUtil::isInteger(const double x, const double tol /*= MathConstant::EPS*/)
{
	double integer;
	const double frac = modf(x, &integer);
	return -tol <= frac && frac <= tol;
	//return std::numeric_limits<T>::is_integer;
}

bool MathUtil::isReal(const double x, const double tol /*= MathConstant::EPS*/)
{
	double integer;
	const double frac = modf(x, &integer);
	return frac < -tol || frac > tol;
	//return !std::numeric_limits<T>::is_integer;
}

std::string MathUtil::toBin(const unsigned long dec)
{  return convert_base_field(dec, 2UL);  }

std::string MathUtil::toOct(const unsigned long dec)
{  return convert_base_field(dec, 8UL);  }

std::string MathUtil::toHex(const unsigned long dec)
{  return convert_base_field(dec, 16UL);  }

}  //  namespace swl
