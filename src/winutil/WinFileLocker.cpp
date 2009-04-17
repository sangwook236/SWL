#include "swl/winutil/WinFileLocker.h"


#if defined(WIN32) && defined(_DEBUG)
void* __cdecl operator new(size_t nSize, const char* lpszFileName, int nLine);
#define new new(__FILE__, __LINE__)
//#pragma comment(lib, "mfc80ud.lib")
#endif


namespace swl {

//-----------------------------------------------------------------------------------
//  WinFileLocker

#if defined(_UNICODE) || defined(UNICODE)
WinFileLocker::WinFileLocker(const std::wstring &filename)
#else
WinFileLocker::WinFileLocker(const std::string &filename)
#endif
: filename_(filename), hFile_(NULL), isLocked_(false)
{
	lock();
}

WinFileLocker::~WinFileLocker()
{
	unlock();
}

bool WinFileLocker::lock()
{
	if (isLocked()) return true;

	hFile_ = CreateFile(
		generateLockFilename().c_str(),
		GENERIC_READ | GENERIC_WRITE, 
		FILE_SHARE_READ | FILE_SHARE_WRITE, 
		NULL, 
		CREATE_ALWAYS, 
		FILE_ATTRIBUTE_NORMAL, 
		NULL
	);
	if (INVALID_HANDLE_VALUE == hFile_)
		return false;
	else
	{
		// lock a file & create a lock file
		if (FALSE == LockFile(hFile_, 0, 0, 0xFFFFFFFF, 0xFFFFFFFF))
		{
			CloseHandle(hFile_);
			hFile_ = NULL;

			return false;
		}
	}

	// we own the lock
	isLocked_ = true;
	return true;
}

bool WinFileLocker::unlock()
{
	if (!isLocked()) return true;

	// unlock the file
	const BOOL isUnlocked = UnlockFile(hFile_, 0, 0, 0xFFFFFFFF, 0xFFFFFFFF);
	CloseHandle(hFile_);
	hFile_ = NULL;

	// delete the lock file
	const BOOL isDeleted = DeleteFile(generateLockFilename().c_str());

	isLocked_ = false;
	return FALSE != isUnlocked && FALSE != isDeleted;
}

#if defined(_UNICODE) || defined(UNICODE)
std::wstring WinFileLocker::generateLockFilename() const
#else
std::string WinFileLocker::generateLockFilename() const
#endif
{
#if defined(_UNICODE) || defined(UNICODE)
	return filename_ + L".lck";
#else
	return filename_ + ".lck"; 
#endif
}

}  // namespace swl
