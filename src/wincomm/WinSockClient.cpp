#include "swl/wincomm/WinSockClient.h"


#if defined(WIN32) && defined(_DEBUG)
void* __cdecl operator new(size_t nSize, const char* lpszFileName, int nLine);
#define new new(__FILE__, __LINE__)
//#pragma comment(lib, "mfc80ud.lib")
#endif


namespace swl {

//-----------------------------------------------------------------------------------
//

/*static*/ bool WinSockClient::initialize()
{
	// initialize Winsock
	WSADATA wsaData;
	if (0 != WSAStartup(MAKEWORD(2, 2), &wsaData)) return false;

	if (LOBYTE(wsaData.wVersion) != 2 || HIBYTE(wsaData.wVersion) != 2)
	{
		WSACleanup();
		return false;
	}

	return true;
}

/*static*/ bool WinSockClient::finalize()
{
	return 0 == WSACleanup();
}

WinSockClient::WinSockClient(const std::string &hostName, const unsigned int portNum)
: socket_(INVALID_SOCKET), isConnected_(false)
{
	// create a SOCKET for connecting to server
	socket_ = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
	if (INVALID_SOCKET == socket_)
		return;

	// the sockaddr_in structure specifies the address family, IP address, and port of the server to be connected to
	sockaddr_in clientService; 
	clientService.sin_family = AF_INET;
	clientService.sin_addr.s_addr = inet_addr(hostName.c_str());
	clientService.sin_port = htons(portNum);

	// connect to server
	if (connect(socket_, (SOCKADDR*)&clientService, sizeof(clientService)) != SOCKET_ERROR)
		isConnected_ = true;
	else
	{
		closesocket(socket_);
		socket_ = INVALID_SOCKET;
	}
}

WinSockClient::~WinSockClient()
{
	if (INVALID_SOCKET != socket_)
	{
		closesocket(socket_);
		socket_ = INVALID_SOCKET;
	}

	isConnected_ = false;
}

size_t WinSockClient::send(const unsigned char *msg, const size_t len) const
{
	if (isConnected())
		return ::send(socket_, (const char *)msg, (int)len, 0);
	else return -1;
}

bool WinSockClient::isConnected() const
{
	return isConnected_ && INVALID_SOCKET != socket_;
}

}  // namespace swl
