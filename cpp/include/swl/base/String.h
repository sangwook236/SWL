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
 *	@brief  string�� ���� utility class.
 *
 *	string�� ���õ� utility API�� �����Ѵ�.
 */
struct SWL_BASE_API String
{
public:
	//typedef String base_type;

public:
	/**
	 *	@brief  unicode string�� multi-byte string���� ��ȯ�ϴ� �Լ�.
	 *	@param[in]  wcstr  ��ȯ�ؾ� �� unicode string.
	 *	@return  unicode string�� ��ȯ�� multi-byte string�� ��ȯ.
	 */
	static std::string wcs2mbs(const std::wstring &wcstr);
	/**
	 *	@brief  multi-byte string�� unicode string���� ��ȯ�ϴ� �Լ�.
	 *	@param[in]  mbstr  ��ȯ�ؾ� �� multi-byte string.
	 *	@return  multi-byte string�� ��ȯ�� unicode string�� ��ȯ.
	 */
	static std::wstring mbs2wcs(const std::string &mbstr);

	/**
	*	@brief  unicode string�� UTF-8 string���� ��ȯ�ϴ� �Լ�.
	*	@param[in]  wcstr  ��ȯ�ؾ� �� unicode string.
	*	@return  unicode string�� ��ȯ�� UTF-8 string�� ��ȯ.
	*/
	static std::string wcs2utf8(const std::wstring &wcstr);
	/**
	*	@brief  UTF-8 string�� unicode string���� ��ȯ�ϴ� �Լ�.
	*	@param[in]  utf8  ��ȯ�ؾ� �� UTF-8 string.
	*	@return  UTF-8 string�� ��ȯ�� unicode string�� ��ȯ.
	*/
	static std::wstring utf82wcs(const std::string &utf8);

	/**
	 *	@brief  10������ ASCII character code�� ��ȯ.
	 *	@param[in]  dec  ��ȯ�ؾ� �� 10����. 0 <= dec <= 9 ������ �ڿ���.
	 *	@return  10������ ��ȯ�� ASCII character code ��ȯ.
	 *
	 *	��ȯ�Ǵ� ASCII code�� ������ '0' <= ascii <= '9' �����̴�.
	 *	���� �߻��� -1�� ��ȯ�Ѵ�.
	 */
	static unsigned char dec2ascii(const unsigned char dec);
	/**
	 *	@brief  ASCII code�� �־��� character�� 10������ ��ȯ.
	 *	@param[in]  ascii  ��ȯ�ؾ� �� character�� ASCII code. '0' <= ascii <= '9' ������ ASCII code.
	 *	@return  ASCII code�� �־��� 10���� character�� �ش��ϴ� �ڿ����� ��ȯ.
	 *
	 *	��ȯ�Ǵ� �ڿ����� ������ 0 <= dec <= 9 �����̴�.
	 *	���� �߻��� -1�� ��ȯ�Ѵ�.
	 */
	static unsigned char ascii2dec(const unsigned char ascii);
	/**
	 *	@brief  16������ ASCII character code�� ��ȯ.
	 *	@param[in]  hex  ��ȯ�ؾ� �� 16����. 0(0x0) <= hex <= 15(0xF) ������ 16����.
	 *	@param[in]  isUpperCase  16���� �� 10(0xA) ~ 15(0xF) ������ ���� �빮���� 'A' ~ 'F'�� ��ȯ�ϴ°��� ����.
	 *	@return  16������ ��ȯ�� ASCII character code ��ȯ.
	 *
	 *	��ȯ�Ǵ� ASCII code�� '0' <= ascii <= '9', 'a' <= ascii <= 'f', �Ǵ� 'A' <= ascii <= 'F' ������ ���̴�.
	 *	���� �߻��� -1�� ��ȯ�Ѵ�.
	 */
	static unsigned char hex2ascii(const unsigned char hex, const bool isUpperCase = true);
	/**
	 *	@brief  ASCII code�� �־��� character�� 16������ ��ȯ.
	 *	@param[in]  ascii  ��ȯ�ؾ� �� character�� ASCII code. '0' <= ascii <= '9', 'a' <= ascii <= 'f', �Ǵ� 'A' <= ascii <= 'F' ������ ASCII code.
	 *	@return  ASCII code�� �־��� 16���� character�� �ش��ϴ� �ڿ����� ��ȯ.
	 *
	 *	��ȯ�Ǵ� �ڿ����� ������ 0(0x0) <= hex <= 15(0xF) �����̴�.
	 *	���� �߻��� -1�� ��ȯ�Ѵ�.
	 */
	static unsigned char ascii2hex(const unsigned char ascii);
};

}  // namespace swl


#endif  // __SWL_BASE__STRING__H_
