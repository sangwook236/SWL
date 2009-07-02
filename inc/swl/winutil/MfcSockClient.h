#if !defined(__SWL_WIN_UTIL__MFC_SOCKET_CLIENT__H_ )
#define __SWL_WIN_UTIL__MFC_SOCKET_CLIENT__H_ 1


#include "swl/winutil/ExportWinUtil.h"
#include <string>

namespace swl {

class CSocket;

//-----------------------------------------------------------------------------------
//

class SWL_WIN_UTIL_API MfcSockClient
{
public:
	MfcSockClient(const std::string &hostName, const unsigned int portNum);
	~MfcSockClient();

public:
	static bool initialize();
	static bool finalize();

public:
	size_t send(const unsigned char *msg, const size_t len) const;

	bool isConnected() const  {  return isConnected_ && NULL != socket_;  }

private:
	CSocket *socket_;
	bool isConnected_;
};

}  // namespace swl


#endif  // __SWL_WIN_UTIL__MFC_SOCKET_CLIENT__H_ 
