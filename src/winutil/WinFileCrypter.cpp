#include "swl/winutil/WinFileCrypter.h"


#if defined(_MSC_VER) && defined(_DEBUG)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//-----------------------------------------------------------------------------------
//  WinFileCrypter

#if defined(_UNICODE) || defined(UNICODE)
WinFileCrypter::WinFileCrypter(const std::wstring &filename, const bool isEcrypted)
#else
WinFileCrypter::WinFileCrypter(const std::string &filename, const bool isEcrypted)
#endif
: filename_(filename), isDone_(false)
{
	if (isEcrypted) encrypt();
	else decrypt();
}

WinFileCrypter::~WinFileCrypter()
{
}

void WinFileCrypter::encrypt()
{
	isDone_ = EncryptFile(filename_.c_str()) != FALSE;
}

void WinFileCrypter::decrypt()
{
	isDone_ = DecryptFile(filename_.c_str(), 0) != FALSE;
}

}  // namespace swl
