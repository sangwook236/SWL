#if !defined(__SWL_UTILITY__STRING_UTIL__H_)
#define __SWL_UTILITY__STRING_UTIL__H_ 1


#include "swl/utility/ExportUtility.h"
#include <string>

namespace swl {

//-----------------------------------------------------------------------------------
//	string utility

/**
 *	@brief  Unicode 문자열과 multi-byte 문자열을 위한 utility class.
 *
 *	unicode string을 multi-byte string으로, multi-byte string을 unicode string으로 변환하는 utility API를 제공한다.
 */
class SWL_UTILITY_API StringUtil
{
public:
	//typedef StringUtil base_type;

public:
	/**
	 *	@brief  unicode string을 multi-byte string으로 변환하는 함수.
	 *	@param  wcstr  [in] 변환해야 할 unicode string.
	 *	@return  unicode string을 변환한 multi-byte string이 반환.
	 */
	static std::string wcs2mbs(const std::wstring &wcstr);
	/**
	 *	@brief  multi-byte string을 unicode string으로 변환하는 함수.
	 *	@param  mbstr  [in] 변환해야 할 multi-byte string.
	 *	@return  multi-byte string을 변환한 unicode string이 반환.
	 */
	static std::wstring mbs2wcs(const std::string &mbstr);
};

}  // namespace swl


#endif  // __SWL_UTILITY__STRING_UTIL__H_
