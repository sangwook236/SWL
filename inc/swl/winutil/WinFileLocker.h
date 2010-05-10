#if !defined(__SWL_WIN_UTIL__WIN_FILE_LOCKER__H_ )
#define __SWL_WIN_UTIL__WIN_FILE_LOCKER__H_ 1


#include "swl/winutil/ExportWinUtil.h"
#include <windows.h>
#include <string>

namespace swl {

//-----------------------------------------------------------------------------------
//  WinFileLocker

/**
 *	@brief  파일 잠금 수행하는 Windows utility class.
 *
 *	파일 잠금 및 잠금 해제를 수행하기 위한 클래스이다.
 */
class SWL_WIN_UTIL_API WinFileLocker
{
	/**
	 *	@brief  [ctor] constructor.
	 *	@param[in]  filename  파일 잠금이 적용될 파일명을 지정.
	 *
	 *	지정된 파일에 대한 잠금을 수행한다.
	 */
public:
#if defined(_UNICODE) || defined(UNICODE)
	WinFileLocker(const std::wstring &filename);
#else
	WinFileLocker(const std::string &filename);
#endif
	/**
	 *	@brief  [dtor] default destructor.
	 *
	 *	지정된 파일에 대한 잠금 해제를 수행한다.
	 */
	~WinFileLocker();

public:
	/**
	 *	@brief  지정된 파일의 잠금을 수행.
	 *	@return  파일 잠금이 성공적으로 수행되었다면 true를 반환.
	 *
	 *	지정된 파일에 대한 잠금을 수행한다. <br>
	 *	파일 잠금이 성공하였을 경우 true를, 그렇지 않을 경우 false를 반환한다.
	 */
	bool lock();
	/**
	 *	@brief  지정된 파일의 잠금 해제를 수행.
	 *	@return  파일 잠금 해제가 성공적으로 수행되었다면 true를 반환.
	 *
	 *	지정된 파일에 대한 잠금 해제를 수행한다. <br>
	 *	파일 잠금 해제가 성공하였을 경우 true를, 그렇지 않을 경우 false를 반환한다.
	 */
	bool unlock();

	/**
	 *	@brief  지정된 파일의 잠금 상태를 반환.
	 *	@return  파일이 잠금 상태라면 true를, 그렇지 않다면 false를 반환.
	 */
	bool isLocked() const  {  return isLocked_;  }

private:
#if defined(_UNICODE) || defined(UNICODE)
	std::wstring generateLockFilename() const;
#else
	std::string generateLockFilename() const;
#endif

private:
#if defined(_UNICODE) || defined(UNICODE)
	const std::wstring filename_;
#else
	const std::string filename_;
#endif
	HANDLE hFile_;

	bool isLocked_;
};

}  // namespace swl


#endif  // __SWL_WIN_UTIL__WIN_FILE_LOCKER__H_ 
