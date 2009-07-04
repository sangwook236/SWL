#include "swl/winutil/MfcSockClient.h"
#include <afxsock.h>


#if defined(_DEBUG)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
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
