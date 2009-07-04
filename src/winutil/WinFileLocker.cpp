#include "swl/Config.h"
#include "swl/winutil/WinFileLocker.h"


#if defined(_DEBUG)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//-----------------------------------------------------------------------------------
//  WinFileLocker

#if defined(_UNICODE) || defined(UNICODE)
WinFileLocker::WinFileLocker(const std::wstring &filename)
#else
WinFileLocker::WinFileLocker(const std::string &filename)
#endif
: filename_(filename), hFile_(INVALID_HANDLE_VALUE), isLocked_(false)
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
			hFile_ = INVALID_HANDLE_VALUE;

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
	hFile_ = INVALID_HANDLE_VALUE;

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
