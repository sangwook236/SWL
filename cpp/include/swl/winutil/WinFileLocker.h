#if !defined(__SWL_WIN_UTIL__WIN_FILE_LOCKER__H_ )
#define __SWL_WIN_UTIL__WIN_FILE_LOCKER__H_ 1


#include "swl/winutil/ExportWinUtil.h"
#include <windows.h>
#include <string>

namespace swl {

//-----------------------------------------------------------------------------------
//  WinFileLocker

/**
 *	@brief  ���� ��� �����ϴ� Windows utility class.
 *
 *	���� ��� �� ��� ������ �����ϱ� ���� Ŭ�����̴�.
 */
class SWL_WIN_UTIL_API WinFileLocker
{
	/**
	 *	@brief  [ctor] constructor.
	 *	@param[in]  filename  ���� ����� ����� ���ϸ��� ����.
	 *
	 *	������ ���Ͽ� ���� ����� �����Ѵ�.
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
	 *	������ ���Ͽ� ���� ��� ������ �����Ѵ�.
	 */
	~WinFileLocker();

public:
	/**
	 *	@brief  ������ ������ ����� ����.
	 *	@return  ���� ����� ���������� ����Ǿ��ٸ� true�� ��ȯ.
	 *
	 *	������ ���Ͽ� ���� ����� �����Ѵ�. <br>
	 *	���� ����� �����Ͽ��� ��� true��, �׷��� ���� ��� false�� ��ȯ�Ѵ�.
	 */
	bool lock();
	/**
	 *	@brief  ������ ������ ��� ������ ����.
	 *	@return  ���� ��� ������ ���������� ����Ǿ��ٸ� true�� ��ȯ.
	 *
	 *	������ ���Ͽ� ���� ��� ������ �����Ѵ�. <br>
	 *	���� ��� ������ �����Ͽ��� ��� true��, �׷��� ���� ��� false�� ��ȯ�Ѵ�.
	 */
	bool unlock();

	/**
	 *	@brief  ������ ������ ��� ���¸� ��ȯ.
	 *	@return  ������ ��� ���¶�� true��, �׷��� �ʴٸ� false�� ��ȯ.
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
