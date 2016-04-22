#if !defined(__SWL_BASE__STRING__H_)
#define __SWL_BASE__STRING__H_ 1


#include "swl/base/ExportBase.h"
#include <string>


namespace swl {

//--------------------------------------------------------------------------
// 

#if defined(_UNICODE) || defined(UNICODE)
typedef wchar_t char_t;
typedef std::wstring string_t;
#define SWL_STR(str) L##str
#else
typedef char char_t;
typedef std::string string_t;
#define SWL_STR(str) str
#endif

//-----------------------------------------------------------------------------------
//	string

/**
 *	@brief  string을 위한 utility class.
 *
 *	string과 관련된 utility API를 제공한다.
 */
struct SWL_BASE_API String
{
public:
	//typedef String base_type;

public:
	/**
	 *	@brief  unicode string을 multi-byte string으로 변환하는 함수.
	 *	@param[in]  wcstr  변환해야 할 unicode string.
	 *	@return  unicode string을 변환한 multi-byte string이 반환.
	 */
	static std::string wcs2mbs(const std::wstring &wcstr);
	/**
	 *	@brief  multi-byte string을 unicode string으로 변환하는 함수.
	 *	@param[in]  mbstr  변환해야 할 multi-byte string.
	 *	@return  multi-byte string을 변환한 unicode string이 반환.
	 */
	static std::wstring mbs2wcs(const std::string &mbstr);

	/**
	 *	@brief  10진수를 ASCII character code로 변환.
	 *	@param[in]  dec  변환해야 할 10진수. 0 <= dec <= 9 사이의 자연수.
	 *	@return  10진수를 변환한 ASCII character code 반환.
	 *
	 *	반환되는 ASCII code의 범위는 '0' <= ascii <= '9' 사이이다.
	 *	오류 발생시 -1을 반환한다.
	 */
	static unsigned char dec2ascii(const unsigned char dec);
	/**
	 *	@brief  ASCII code로 주어진 character를 10진수로 변환.
	 *	@param[in]  ascii  변환해야 할 character의 ASCII code. '0' <= ascii <= '9' 사이의 ASCII code.
	 *	@return  ASCII code로 주어진 10진수 character에 해당하는 자연수를 반환.
	 *
	 *	반환되는 자연수의 범위는 0 <= dec <= 9 사이이다.
	 *	오류 발생시 -1을 반환한다.
	 */
	static unsigned char ascii2dec(const unsigned char ascii);
	/**
	 *	@brief  16진수를 ASCII character code로 변환.
	 *	@param[in]  hex  변환해야 할 16진수. 0(0x0) <= hex <= 15(0xF) 사이의 16진수.
	 *	@param[in]  isUpperCase  16진수 중 10(0xA) ~ 15(0xF) 사이의 값이 대문자인 'A' ~ 'F'로 변환하는가를 지정.
	 *	@return  16진수를 변환한 ASCII character code 반환.
	 *
	 *	반환되는 ASCII code는 '0' <= ascii <= '9', 'a' <= ascii <= 'f', 또는 'A' <= ascii <= 'F' 사이의 값이다.
	 *	오류 발생시 -1을 반환한다.
	 */
	static unsigned char hex2ascii(const unsigned char hex, const bool isUpperCase = true);
	/**
	 *	@brief  ASCII code로 주어진 character를 16진수로 변환.
	 *	@param[in]  ascii  변환해야 할 character의 ASCII code. '0' <= ascii <= '9', 'a' <= ascii <= 'f', 또는 'A' <= ascii <= 'F' 사이의 ASCII code.
	 *	@return  ASCII code로 주어진 16진수 character에 해당하는 자연수를 반환.
	 *
	 *	반환되는 자연수의 범위는 0(0x0) <= hex <= 15(0xF) 사이이다.
	 *	오류 발생시 -1을 반환한다.
	 */
	static unsigned char ascii2hex(const unsigned char ascii);
};

}  // namespace swl


#endif  // __SWL_BASE__STRING__H_
