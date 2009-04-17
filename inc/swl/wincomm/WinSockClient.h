#if !defined(__SWL_WIN_COMM__WIN_SOCKET_CLIENT__H_ )
#define __SWL_WIN_COMM__WIN_SOCKET_CLIENT__H_ 1


#include "swl/wincomm/ExportWinComm.h"
#include <winsock2.h>
#include <string>

namespace swl {

//-----------------------------------------------------------------------------------
//

class SWL_WIN_COMM_API WinSockClient
{
public:
	WinSockClient(const std::string &hostName, const unsigned int portNum);
	~WinSockClient();

public:
	static bool initialize();
	static bool finalize();

public:
	size_t send(const unsigned char *msg, const size_t len) const;

	bool isConnected() const;

private:
	SOCKET socket_;
	bool isConnected_;
};

}  // namespace swl


#endif  // __SWL_WIN_COMM__WIN_SOCKET_CLIENT__H_ 
