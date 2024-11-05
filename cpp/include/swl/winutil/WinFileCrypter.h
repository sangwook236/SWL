#if !defined(__SWL_WIN_UTIL__WIN_FILE_CRYPTER__H_ )
#define __SWL_WIN_UTIL__WIN_FILE_CRYPTER__H_ 1


#include "swl/winutil/ExportWinUtil.h"
#include <windows.h>
#include <string>

namespace swl {

//-----------------------------------------------------------------------------------
//  WinFileCrypter

/**
 *	@brief  파일 암호화/복호화를 수행하는 Windows utility class.
 *
 *	파일 암호화 및 복호화를 수행하기 위한 클래스이다.
 */
class SWL_WIN_UTIL_API WinFileCrypter
{
	/**
	 *	@brief  [ctor] constructor.
	 *	@param[in]  filename  파일 암호화 및 복호화가 적용될 파일명을 지정.
	 *	@param[in]  isEcrypted  파일을 암호화할 것인지 복호화할 것인지 지정.
	 */
public:
#if defined(_UNICODE) || defined(UNICODE)
	WinFileCrypter(const std::wstring &filename, const bool isEcrypted);
#else
	WinFileCrypter(const std::string &filename, const bool isEcrypted);
#endif
	/**
	 *	@brief  [dtor] default destructor.
	 */
	~WinFileCrypter();

public:
	/**
	 *	@brief  지정된 파일의 암호화 및 복호화 상태를 반환.
	 *	@return  파일이 정상적으로 암호화 및 복호화된 상태라면 true를, 그렇지 않다면 false를 반환.
	 */
	bool isDone() const  {  return isDone_;  }

private:
	/**
	 *	@brief  지정된 파일의 암호화를 수행.
	 *	@return  파일 암호화가 성공적으로 수행되었다면 true를 반환.
	 *
	 *	지정된 파일에 대한 암호화를 수행한다. <br>
	 *	파일 암호화가 성공하였을 경우 true를, 그렇지 않을 경우 false를 반환한다.
	 */
	void encrypt();
	/**
	 *	@brief  지정된 파일의 복호화를 수행.
	 *	@return  파일 복호화가 성공적으로 수행되었다면 true를 반환.
	 *
	 *	지정된 파일에 대한 복호화를 수행한다. <br>
	 *	파일 복호화가 성공하였을 경우 true를, 그렇지 않을 경우 false를 반환한다.
	 */
	void decrypt();

private:
#if defined(_UNICODE) || defined(UNICODE)
	const std::wstring filename_;
#else
	const std::string filename_;
#endif

	bool isDone_;
};

}  // namespace swl


#endif  // __SWL_WIN_UTIL__WIN_FILE_CRYPTER__H_ 
