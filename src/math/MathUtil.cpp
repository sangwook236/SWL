#include "swl/Config.h"
#include "swl/math/MathUtil.h"
#include <cmath>
#include <algorithm>


#if defined(_DEBUG)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
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
	unsigned long dec2 = 0ul;
	bool isNegative;
	if (dec == 0ul)
#if defined(_UNICODE) || defined(UNICODE)
		return L"0";
#else
		return "0";
#endif
	else if (dec > 0ul)
	{
		dec2 = dec;
		isNegative = false;
	}
	else
	{
		dec2 = -dec;
		isNegative = true;
	}

#if defined(_UNICODE) || defined(UNICODE)
	std::wstring num;
#else
	std::string num;
#endif
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
	if (isNegative) num += L'-';
#else
	num += std::string::value_type(dec2 >= 10ul ? 'A' + dec2 - 10ul : '0' + dec2);
	if (isNegative) num += '-';
#endif
	std::reverse(num.begin(), num.end());

	return num;
}

#if defined(_UNICODE) || defined(UNICODE)
static long convert_base_field(const std::wstring &num, const unsigned long base)
#else
static long convert_base_field(const std::string &num, const unsigned long base)
#endif
{
	if (num.empty()) return 0ul;

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

		// TODO [check] >>
		return (unsigned long)-1;
	}

	const size_t len = num.length();
	unsigned long dec = 0ul;
	unsigned long digit;
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
				return (unsigned long)-1;
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
				return (unsigned long)-1;
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
				return (unsigned long)-1;
			break;
		default:
			return (unsigned long)-1;
		}

		dec += digit * (long)std::pow(double(base), double(len - idx - 1ul));
	}

	return isNegative ? -dec : dec;
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
/*static*/ std::wstring MathUtil::dec2bin(const unsigned long dec)
#else
/*static*/ std::string MathUtil::dec2bin(const unsigned long dec)
#endif
{  return convert_base_field(dec, 2ul);  }

#if defined(_UNICODE) || defined(UNICODE)
/*static*/ std::wstring MathUtil::dec2oct(const unsigned long dec)
#else
/*static*/ std::string MathUtil::dec2oct(const unsigned long dec)
#endif
{  return convert_base_field(dec, 8ul);  }

#if defined(_UNICODE) || defined(UNICODE)
/*static*/ std::wstring MathUtil::dec2hex(const unsigned long dec)
#else
/*static*/ std::string MathUtil::dec2hex(const unsigned long dec)
#endif
{  return convert_base_field(dec, 16ul);  }

#if defined(_UNICODE) || defined(UNICODE)
/*static*/ unsigned long MathUtil::bin2dec(const std::wstring &bin)
#else
/*static*/ unsigned long MathUtil::bin2dec(const std::string &bin)
#endif
{  return convert_base_field(bin, 2ul);  }

#if defined(_UNICODE) || defined(UNICODE)
/*static*/ unsigned long MathUtil::oct2dec(const std::wstring &oct)
#else
/*static*/ unsigned long MathUtil::oct2dec(const std::string &oct)
#endif
{  return convert_base_field(oct, 8ul);  }

#if defined(_UNICODE) || defined(UNICODE)
/*static*/ unsigned long MathUtil::hex2dec(const std::wstring &hex)
#else
/*static*/ unsigned long MathUtil::hex2dec(const std::string &hex)
#endif
{  return convert_base_field(hex, 16ul);  }

}  //  namespace swl
