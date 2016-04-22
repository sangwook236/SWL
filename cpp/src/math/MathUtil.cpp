#include "swl/Config.h"
#include "swl/math/MathUtil.h"
#include <cmath>
#include <algorithm>
#include <limits>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif

#if defined(PI)  // for djgpp
#	undef PI
#endif


namespace swl {

namespace {
namespace local {

#if defined(_UNICODE) || defined(UNICODE)
static std::wstring convert_base_field(const long dec, const long base)
#else
static std::string convert_base_field(const long dec, const long base)
#endif
{
    long dec2 = 0;
	bool isNegative;
	if (dec == 0)
#if defined(_UNICODE) || defined(UNICODE)
		return L"0";
#else
		return "0";
#endif
	else if (dec > 0)
	{
		dec2 = dec;
		isNegative = false;
	}
	else
	{
		// TODO [check] >>
		dec2 = -dec;
		isNegative = true;
	}

#if defined(_UNICODE) || defined(UNICODE)
	std::wstring num;
#else
	std::string num;
#endif
    long remainder;
	while (dec2 >= base)
	{
		remainder = dec2 % base;
#if defined(_UNICODE) || defined(UNICODE)
		num += std::wstring::value_type(remainder >= 10l ? L'A' + remainder - 10l : L'0' + remainder);
#else
		num += std::string::value_type(remainder >= 10l ? 'A' + remainder - 10l : '0' + remainder);
#endif
		dec2 /= base;
	}
#if defined(_UNICODE) || defined(UNICODE)
	num += std::wstring::value_type(dec2 >= 10l ? L'A' + dec2 - 10l : L'0' + dec2);
	if (isNegative) num += L'-';
#else
	num += std::string::value_type(dec2 >= 10l ? 'A' + dec2 - 10l : '0' + dec2);
	if (isNegative) num += '-';
#endif
	std::reverse(num.begin(), num.end());

	return num;
}

#if defined(_UNICODE) || defined(UNICODE)
static long convert_base_field(const std::wstring &num, const long base)
#else
static long convert_base_field(const std::string &num, const long base)
#endif
{
	if (num.empty()) return 0;

	size_t idx = 0;
	bool isNegative = false;
#if defined(_UNICODE) || defined(UNICODE)
	if (num[idx] == L'+')
#else
	if (num[idx] == '+')
#endif
		++idx;
#if defined(_UNICODE) || defined(UNICODE)
	else if (num[idx] == L'-')
#else
	else if (num[idx] == '-')
#endif
	{
		isNegative = true;
		++idx;

		// FIXME [check] >>
		return std::numeric_limits<long>::max();
	}

	const size_t len = num.length();
	long dec = 0l;
    long digit;
	for (; idx < len; ++idx)
	{
		switch (base)
		{
		case 2:
#if defined(_UNICODE) || defined(UNICODE)
			if (L'0' <= num[idx] && num[idx] <= L'1')
				digit = num[idx] - L'0';
#else
			if ('0' <= num[idx] && num[idx] <= '1')
				digit = num[idx] - '0';
#endif
			else
                // FIXME [check] >>
				return std::numeric_limits<long>::max();
			break;
		case 8:
#if defined(_UNICODE) || defined(UNICODE)
			if (L'0' <= num[idx] && num[idx] <= L'7')
				digit = num[idx] - L'0';
#else
			if ('0' <= num[idx] && num[idx] <= '7')
				digit = num[idx] - '0';
#endif
			else
                // FIXME [check] >>
				return std::numeric_limits<long>::max();
			break;
		case 16:
#if defined(_UNICODE) || defined(UNICODE)
			if (L'0' <= num[idx] && num[idx] <= L'9')
				digit = num[idx] - L'0';
			else if (L'a' <= num[idx] && num[idx] <= L'f')
				digit = num[idx] - L'a' + 10ul;
			else if (L'A' <= num[idx] && num[idx] <= L'F')
				digit = num[idx] - L'A' + 10ul;
#else
			if ('0' <= num[idx] && num[idx] <= '9')
				digit = num[idx] - '0';
			else if ('a' <= num[idx] && num[idx] <= 'f')
				digit = num[idx] - 'a' + 10ul;
			else if ('A' <= num[idx] && num[idx] <= 'F')
				digit = num[idx] - 'A' + 10ul;
#endif
			else
                // FIXME [check] >>
				return std::numeric_limits<long>::max();
			break;
		default:
            // FIXME [check] >>
            return std::numeric_limits<long>::max();
		}

		dec += digit * (long)std::pow(double(base), double(len - idx - 1l));
	}

	// TODO [check] >>
	return isNegative ? -dec : dec;
}

}  // namespace local
}  // unnamed namespace

//-----------------------------------------------------------------------------------------
// struct MathUtil

//        |                     | [-1.0, 1.4)  [1.4, 3.8)  [-3.4, -1.0)  [-5.8, -3.4)
//  value | dist wrt lower      |        -1.0        +1.4          -3.4          -5.8
// ----------------------------------------------------------------------------------------
//  -10.7 | 2.3 = -10.7 - -13.0 |         1.3         3.7          -1.1          -3.5
//  -10.6 | 0.0 = -10.6 - -10.6 |        -1.0         1.4          -3.4          -5.8
//  -10.5 | 0.1 = -10.5 - -10.6 |        -0.9         1.5          -3.3          -5.7
//   13.3 | 2.3 = 13.3 - 11.0   |         1.3         3.7          -1.1          -3.5
//   13.4 | 0.0 = 13.4 - 13.4   |        -1.0         1.4          -3.4          -5.8
//   13.5 | 0.1 = 13.5 - 13.4   |        -0.9         1.5          -3.3          -5.7
/*static*/ double MathUtil::wrap(const double x, const double lower, const double upper, const double tol /*= MathConstant::EPS*/)
{
	const double span = upper - lower;
	const double y = x - lower;

	double intpart;
	const double fractpart = std::modf(y / span, &intpart);
	if (y >= 0.0)
		return x - intpart * span;
	else
		return x - ((fractpart < -tol) ? (intpart - 1.0) : intpart) * span;
}

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

/*static*/ int MathUtil::toPrecedingOdd(const double num)
{
	return 2 * (int)std::floor((num + 1) * 0.5) - 1;
}

/*static*/ int MathUtil::toFollowingOdd(const double num)
{
	return 2 * (int)std::ceil((num + 1) * 0.5) - 1;
}

/*static*/ int MathUtil::toPrecedingEven(const double num)
{
	return 2 * (int)std::floor(num * 0.5);
}

/*static*/ int MathUtil::toFollowingEven(const double num)
{
	return 2 * (int)std::ceil(num * 0.5);
}

#if defined(_UNICODE) || defined(UNICODE)
/*static*/ std::wstring MathUtil::dec2bin(const long dec)
#else
/*static*/ std::string MathUtil::dec2bin(const long dec)
#endif
{  return local::convert_base_field(dec, 2l);  }

#if defined(_UNICODE) || defined(UNICODE)
/*static*/ std::wstring MathUtil::dec2oct(const long dec)
#else
/*static*/ std::string MathUtil::dec2oct(const long dec)
#endif
{  return local::convert_base_field(dec, 8u);  }

#if defined(_UNICODE) || defined(UNICODE)
/*static*/ std::wstring MathUtil::dec2hex(const long dec)
#else
/*static*/ std::string MathUtil::dec2hex(const long dec)
#endif
{  return local::convert_base_field(dec, 16l);  }

#if defined(_UNICODE) || defined(UNICODE)
/*static*/ long MathUtil::bin2dec(const std::wstring &bin)
#else
/*static*/ long MathUtil::bin2dec(const std::string &bin)
#endif
{  return local::convert_base_field(bin, 2l);  }

#if defined(_UNICODE) || defined(UNICODE)
/*static*/ long MathUtil::oct2dec(const std::wstring &oct)
#else
/*static*/ long MathUtil::oct2dec(const std::string &oct)
#endif
{  return local::convert_base_field(oct, 8l);  }

#if defined(_UNICODE) || defined(UNICODE)
/*static*/ long MathUtil::hex2dec(const std::wstring &hex)
#else
/*static*/ long MathUtil::hex2dec(const std::string &hex)
#endif
{  return local::convert_base_field(hex, 16l);  }

}  //  namespace swl
