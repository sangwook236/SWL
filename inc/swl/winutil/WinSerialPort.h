#if !defined(__SWL_WIN_UTIL__SERIAL_PORT__H_)
#define __SWL_WIN_UTIL__SERIAL_PORT__H_ 1


#include "swl/winutil/ExportWinUtil.h"
#include <windows.h>

namespace swl {

template<class T> class GuardedBuffer;
typedef GuardedBuffer<unsigned char> GuardedByteBuffer;

//-----------------------------------------------------------------------------------
//	serial port

class SWL_WIN_UTIL_API WinSerialPort
{
public:
	enum EState { E_ERROR = 0, E_OK, E_TIMEOUT };

public:
	WinSerialPort();
	~WinSerialPort();

public:
#if defined(_UNICODE) || defined(__UNICODE)
	bool connect(const wchar_t *portName, const int baudRate, const size_t inQueueSize, const size_t outQueueSize);
#else
	bool connect(const char *portName, const int baudRate, const size_t inQueueSize, const size_t outQueueSize);
#endif
	void disconnect();

	EState send(const unsigned char *data, const size_t dataSize, const unsigned long timeoutInterval_msec = 100);
	EState receive(GuardedByteBuffer &recvBuf, const unsigned long timeoutInterval_msec = 100, const size_t bufferLen = 0);

	bool cancelIo();

private:
	EState waitFor(OVERLAPPED &ovWait, const unsigned long timeoutInterval, const bool isSending, size_t &transferredBytes);
	void clearError();

private:
	HANDLE hComPort_;
	OVERLAPPED ovSend_, ovRecv_;
};

}  // namespace swl


#endif  // __SWL_WIN_UTIL__SERIAL_PORT__H_
