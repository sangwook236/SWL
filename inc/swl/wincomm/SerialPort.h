#if !defined(__SWL_WIN_COMM__SERIAL_PORT__H_)
#define __SWL_WIN_COMM__SERIAL_PORT__H_ 1


#include "swl/wincomm/ExportWinComm.h"
#include <windows.h>

namespace swl {

class ByteBuffer;

//-----------------------------------------------------------------------------------
//	serial port

class SWL_WIN_COMM_API SerialPort
{
public:
	enum EState { E_ERROR = 0, E_OK, E_TIMEOUT };

public:
	SerialPort();
	~SerialPort();

public:
#if defined(_UNICODE) || defined(__UNICODE)
	bool connect(const wchar_t* portName, const int baudRate, const size_t inQueueSize, const size_t outQueueSize);
#else
	bool connect(const char* portName, const int baudRate, const size_t inQueueSize, const size_t outQueueSize);
#endif
	void disconnect();

	EState send(const unsigned char* data, const size_t dataSize, const unsigned long timeoutInterval_msec = 100);
	EState receive(ByteBuffer& recvBuf, const unsigned long timeoutInterval_msec = 100, const size_t bufferLen = 0);

	bool cancelIo();

private:
	EState waitFor(OVERLAPPED& ovWait, const unsigned long timeoutInterval, const bool isSending, size_t& transferredBytes);
	void clearError();

private:
	HANDLE hComPort_;
	OVERLAPPED ovSend_, ovRecv_;
};

}  // namespace swl


#endif  // __SWL_WIN_COMM__SERIAL_PORT__H_
