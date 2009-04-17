#include "swl/wincomm/MfcSockClient.h"
#include <afxsock.h>


#if defined(WIN32) && defined(_DEBUG)
void* __cdecl operator new(size_t nSize, const char* lpszFileName, int nLine);
#define new new(__FILE__, __LINE__)
//#pragma comment(lib, "mfc80ud.lib")
#endif


namespace swl {

//-----------------------------------------------------------------------------------
//

/*static*/ bool MfcSockClient::initialize()
{
	return 0 == AfxSocketInit();
}

/*static*/ bool MfcSockClient::initialize()
{
	// do nothing
	return true;
}

MfcSockClient::MfcSockClient(const std::string &hostName, const unsigned int portNum)
: socket_(NULL), isConnected_(false)
{
	socket_ = new CSocket();
	socket_->Create();

	if (socket_ && socket_->Connect(CString(hostName.c_str()), portNum))
		isConnected_ = true;
	else
	{
		delete socket_;
		socket_ = NULL;
	}
}

MfcSockClient::~MfcSockClient()
{
	if (isConnected())
		socket_->Close();

	if (socket_)
	{
		delete socket_;
		socket_ = NULL;
	}

	isConnected_ = false;
}

size_t MfcSockClient::send(const unsigned char *msg, const size_t len) const
{
	if (isConnected())
		return socket_->Send(msg, len);
	else return -1;
}

}  // namespace swl
