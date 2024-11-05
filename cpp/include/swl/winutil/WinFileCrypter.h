#if !defined(__SWL_WIN_UTIL__WIN_FILE_CRYPTER__H_ )
#define __SWL_WIN_UTIL__WIN_FILE_CRYPTER__H_ 1


#include "swl/winutil/ExportWinUtil.h"
#include <windows.h>
#include <string>

namespace swl {

//-----------------------------------------------------------------------------------
//  WinFileCrypter

/**
 *	@brief  ���� ��ȣȭ/��ȣȭ�� �����ϴ� Windows utility class.
 *
 *	���� ��ȣȭ �� ��ȣȭ�� �����ϱ� ���� Ŭ�����̴�.
 */
class SWL_WIN_UTIL_API WinFileCrypter
{
	/**
	 *	@brief  [ctor] constructor.
	 *	@param[in]  filename  ���� ��ȣȭ �� ��ȣȭ�� ����� ���ϸ��� ����.
	 *	@param[in]  isEcrypted  ������ ��ȣȭ�� ������ ��ȣȭ�� ������ ����.
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
	 *	@brief  ������ ������ ��ȣȭ �� ��ȣȭ ���¸� ��ȯ.
	 *	@return  ������ ���������� ��ȣȭ �� ��ȣȭ�� ���¶�� true��, �׷��� �ʴٸ� false�� ��ȯ.
	 */
	bool isDone() const  {  return isDone_;  }

private:
	/**
	 *	@brief  ������ ������ ��ȣȭ�� ����.
	 *	@return  ���� ��ȣȭ�� ���������� ����Ǿ��ٸ� true�� ��ȯ.
	 *
	 *	������ ���Ͽ� ���� ��ȣȭ�� �����Ѵ�. <br>
	 *	���� ��ȣȭ�� �����Ͽ��� ��� true��, �׷��� ���� ��� false�� ��ȯ�Ѵ�.
	 */
	void encrypt();
	/**
	 *	@brief  ������ ������ ��ȣȭ�� ����.
	 *	@return  ���� ��ȣȭ�� ���������� ����Ǿ��ٸ� true�� ��ȯ.
	 *
	 *	������ ���Ͽ� ���� ��ȣȭ�� �����Ѵ�. <br>
	 *	���� ��ȣȭ�� �����Ͽ��� ��� true��, �׷��� ���� ��� false�� ��ȯ�Ѵ�.
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
