#if !defined(__SWL_WIN_UTIL__SERIAL_PORT__H_)
#define __SWL_WIN_UTIL__SERIAL_PORT__H_ 1


#include "swl/winutil/ExportWinUtil.h"
#include <windows.h>

namespace swl {

template<class T> class GuardedBuffer;
typedef GuardedBuffer<unsigned char> GuardedByteBuffer;

//-----------------------------------------------------------------------------------
//	serial port

/**
 *	@brief  serial 통신을 지원하기 위한 class.
 *
 *	serial 통신을 위해 필요한 설정은 아래와 같다.
 *		- 통신 port: connect() 함수에서 지정
 *		- Baud rate: connect() 함수에서 지정
 *		- data bit: 8 bits
 *		- stop bit: 1 bit
 *		- parity: none
 *		- H/W handshaking: 사용 안함
 *
 *	내부적으로 asynchronous I/O를 사용하고 있으므로 이 class를 사용하는 S/W에 부담을 적게 주는 장점이 있다.
 *
 *	serial 통신을 통해 data를 전송할 경우에는 send() 함수를 해당 시점에 직접 호출하면 되고,
 *	data 수신의 경우에는 serial 통신을 위해 thread를 열고 해당 thread 안에서 receive() 함수를 호출하면 된다.
 */
class SWL_WIN_UTIL_API WinSerialPort
{
public:
	/**
	 *	@brief  serial 통신 과정에 발생하는 통신 channel의 상태를 지정.
	 *
	 *	통신 channel 상태:
	 *		- E_ERROR: error가 발생한 경우
	 *		- E_OK: 송수신이 정상적으로 이루어진 경우
	 *		- E_TIMEOUT: 지정된 time-out 시간 동안 송수신을 완료하지 못한 경우
	 */
	enum EState { E_ERROR = 0, E_OK, E_TIMEOUT };

public:
	/**
	 *	@brief  [ctor] default contructor.
	 *
	 *	serial 통신을 위해 필요한 설정값들을 초기화한다.
	 */
	WinSerialPort();
	/**
	 *	@brief  [dtor] default destructor.
	 *
	 *	serial 통신을 종료하기 위해 필요한 절차를 수행한다.
	 *
	 *	통신 port가 열려 있는 경우 disconnect() 함수를 호출하여 이를 닫는다.
	 */
	~WinSerialPort();

public:
	/**
	 *	@brief  지정된 COM port와 Baud rate로 serial 통신 channel을 연결.
	 *	@param[in]  portName  serial 통신을 위한 port 이름.
	 *	@param[in]  baudRate  통신을 위해 사용하고자 하는 속도.
	 *	@param[in]  inQueueSize  serial 통신 과정에서 사용하게 되는 input queue의 크기.
	 *	@param[in]  outQueueSize  serial 통신 과정에서 사용하게 되는 output queue의 크기.
	 *	@return  serial 통신 channel이 정상적으로 연결되었다면 true 반환.
	 *
	 *	인자로 넘겨진 port 이름과 Baud rate를 이용하여 통신 채널을 연결하고
	 *	asynchronous I/O 작업을 수행하기 위한 초기화 작업을 수행한다.
	 */
#if defined(_UNICODE) || defined(__UNICODE)
	bool connect(const wchar_t *portName, const int baudRate, const size_t inQueueSize, const size_t outQueueSize);
#else
	bool connect(const char *portName, const int baudRate, const size_t inQueueSize, const size_t outQueueSize);
#endif
	/**
	 *	@brief  serial 통신을 위한 port를 닫음.
	 *
	 *	serial 통신을 위해 연결하였던 channel을 끊고, 활용한 resource를 반환한다.
	 */
	void disconnect();

	/**
	 *	@brief  지정된 data를 연결된 serial 통신 channel을 통해 전송.
	 *	@param[in]  data  전송할 data를 지정하는 pointer.
	 *	@param[in]  dataSize  전송할 data 크기.
	 *	@param[in]  timeoutInterval_msec  data 전송시 대기할 time-out interval. [msec] 단위로 지정.
	 *	@return  통신 channel의 결과 상태를 반환.
	 *
	 *	요청된 data를 serial 통신을 통해 전송한다.
	 *
	 *	asynchronous I/O를 통해 data를 전송하며 이 과정에서 발생하는 channel 상태를 EState 값으로 반환한다.
	 *	EState::E_TIMEOUT의 경우 인자로 지정된 timeoutInterval_msec 시간을 초과하였을 경우 반환된다.
	 */
	EState send(const unsigned char *data, const size_t dataSize, const unsigned long timeoutInterval_msec = 100);
	/**
	 *	@brief  연결된 serial 통신 channel을 통해 data를 수신.
	 *	@param[out]  recvBuf  수신된 data를 저장한 receive data buffer.
	 *	@param[in]  timeoutInterval_msec  data 수신을 위해 대기할 time-out interval. [msec] 단위로 지정.
	 *	@param[in]  bufferLen  asynchronous I/O를 통해 data를 수신할 때 수신 buffer의 크기를 지정. 0으로 지정된다면 해당 buffer의 크기는 8192 크기로 할당.
	 *	@return  통신 channel의 결과 상태를 반환.
	 *
	 *	serial 통신을 통해 수신되는 data를 인자로 지정된 ByteBuffer class의 객체에 저장한다.
	 *
	 *	asynchronous I/O를 통해 data를 수신하며 이 과정에서 발생하는 channel 상태를 EState 값으로 반환한다.
	 *	EState::E_TIMEOUT의 경우 인자로 지정된 timeoutInterval_msec 시간을 초과하였을 경우 반환된다.
	 */
	EState receive(GuardedByteBuffer &recvBuf, const unsigned long timeoutInterval_msec = 100, const size_t bufferLen = 0);

	/**
	 *	@brief  수행 중인 I/O 작업을 취소시킴.
	 *	@return  I/O 작업이 정상적으로 수행되었다면 true를 반환.
	 */
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
