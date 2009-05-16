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

#if defined(_UNICODE) || defined(UNICODE)
static std::wstring convert_base_field(const unsigned long dec, const unsigned long base)
#else
static std::string convert_base_field(const unsigned long dec, const unsigned long base)
#endif
{
#if defined(_UNICODE) || defined(UNICODE)
	std::wstring num;
#else
	std::string num;
#endif
	unsigned long dec2 = dec;
	unsigned long remainder;
	while (dec2 >= base)
	{
		remainder = dec2 % base;
#if defined(_UNICODE) || defined(UNICODE)
		num += std::wstring::value_type(remainder >= 10ul ? L'A' + remainder - 10ul : L'0' + remainder);
#else
		num += std::string::value_type(remainder >= 10ul ? 'A' + remainder - 10ul : '0' + remainder);
#endif
		dec2 /= base;
	}
#if defined(_UNICODE) || defined(UNICODE)
	num += std::wstring::value_type(dec2 >= 10ul ? L'A' + dec2 - 10ul : L'0' + dec2);
#else
	num += std::string::value_type(dec2 >= 10ul ? 'A' + dec2 - 10ul : '0' + dec2);
#endif
	std::reverse(num.begin(), num.end());
	return num;
}

}  // unnamed namespace

//-----------------------------------------------------------------------------------------
// struct MathUtil

/*static*/ bool MathUtil::isInteger(const double x, const double tol /*= MathConstant::EPS*/)
{
	double integer;
	const double frac = std::modf(x, &integer);
	return -tol <= frac && frac <= tol;
	//return std::numeric_limits<T>::is_integer;
}

/*static*/ bool MathUtil::isReal(const double x, const double tol /*= MathConstant::EPS*/)
{
	double integer;
	const double frac = std::modf(x, &integer);
	return frac < -tol || frac > tol;
	//return !std::numeric_limits<T>::is_integer;
}

#if defined(_UNICODE) || defined(UNICODE)
/*static*/ std::wstring MathUtil::toBin(const unsigned long dec)
#else
/*static*/ std::string MathUtil::toBin(const unsigned long dec)
#endif
{  return convert_base_field(dec, 2UL);  }

#if defined(_UNICODE) || defined(UNICODE)
/*static*/ std::wstring MathUtil::toOct(const unsigned long dec)
#else
/*static*/ std::string MathUtil::toOct(const unsigned long dec)
#endif
{  return convert_base_field(dec, 8UL);  }

#if defined(_UNICODE) || defined(UNICODE)
/*static*/ std::wstring MathUtil::toHex(const unsigned long dec)
#else
/*static*/ std::string MathUtil::toHex(const unsigned long dec)
#endif
{  return convert_base_field(dec, 16UL);  }

}  //  namespace swl
