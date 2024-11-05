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
 *	@brief  serial ����� �����ϱ� ���� class.
 *
 *	serial ����� ���� �ʿ��� ������ �Ʒ��� ����.
 *		- ��� port: connect() �Լ����� ����
 *		- Baud rate: connect() �Լ����� ����
 *		- data bit: 8 bits
 *		- stop bit: 1 bit
 *		- parity: none
 *		- H/W handshaking: ��� ����
 *
 *	���������� asynchronous I/O�� ����ϰ� �����Ƿ� �� class�� ����ϴ� S/W�� �δ��� ���� �ִ� ������ �ִ�.
 *
 *	serial ����� ���� data�� ������ ��쿡�� send() �Լ��� �ش� ������ ���� ȣ���ϸ� �ǰ�,
 *	data ������ ��쿡�� serial ����� ���� thread�� ���� �ش� thread �ȿ��� receive() �Լ��� ȣ���ϸ� �ȴ�.
 */
class SWL_WIN_UTIL_API WinSerialPort
{
public:
	/**
	 *	@brief  serial ��� ������ �߻��ϴ� ��� channel�� ���¸� ����.
	 *
	 *	��� channel ����:
	 *		- E_ERROR: error�� �߻��� ���
	 *		- E_OK: �ۼ����� ���������� �̷���� ���
	 *		- E_TIMEOUT: ������ time-out �ð� ���� �ۼ����� �Ϸ����� ���� ���
	 */
	enum EState { E_ERROR = 0, E_OK, E_TIMEOUT };

public:
	/**
	 *	@brief  [ctor] default constructor.
	 *
	 *	serial ����� ���� �ʿ��� ���������� �ʱ�ȭ�Ѵ�.
	 */
	WinSerialPort();
	/**
	 *	@brief  [dtor] default destructor.
	 *
	 *	serial ����� �����ϱ� ���� �ʿ��� ������ �����Ѵ�.
	 *
	 *	��� port�� ���� �ִ� ��� disconnect() �Լ��� ȣ���Ͽ� �̸� �ݴ´�.
	 */
	~WinSerialPort();

public:
	/**
	 *	@brief  ������ COM port�� Baud rate�� serial ��� channel�� ����.
	 *	@param[in]  portName  serial ����� ���� port �̸�.
	 *	@param[in]  baudRate  ����� ���� ����ϰ��� �ϴ� �ӵ�.
	 *	@param[in]  inQueueSize  serial ��� �������� ����ϰ� �Ǵ� input queue�� ũ��.
	 *	@param[in]  outQueueSize  serial ��� �������� ����ϰ� �Ǵ� output queue�� ũ��.
	 *	@return  serial ��� channel�� ���������� ����Ǿ��ٸ� true ��ȯ.
	 *
	 *	���ڷ� �Ѱ��� port �̸��� Baud rate�� �̿��Ͽ� ��� ä���� �����ϰ�
	 *	asynchronous I/O �۾��� �����ϱ� ���� �ʱ�ȭ �۾��� �����Ѵ�.
	 */
#if defined(_UNICODE) || defined(__UNICODE)
	bool connect(const wchar_t *portName, const int baudRate, const size_t inQueueSize, const size_t outQueueSize);
#else
	bool connect(const char *portName, const int baudRate, const size_t inQueueSize, const size_t outQueueSize);
#endif
	/**
	 *	@brief  serial ����� ���� port�� ����.
	 *
	 *	serial ����� ���� �����Ͽ��� channel�� ����, Ȱ���� resource�� ��ȯ�Ѵ�.
	 */
	void disconnect();

	/**
	 *	@brief  ������ data�� ����� serial ��� channel�� ���� ����.
	 *	@param[in]  data  ������ data�� �����ϴ� pointer.
	 *	@param[in]  dataSize  ������ data ũ��.
	 *	@param[in]  timeoutInterval_msec  data ���۽� ����� time-out interval. [msec] ������ ����.
	 *	@return  ��� channel�� ��� ���¸� ��ȯ.
	 *
	 *	��û�� data�� serial ����� ���� �����Ѵ�.
	 *
	 *	asynchronous I/O�� ���� data�� �����ϸ� �� �������� �߻��ϴ� channel ���¸� EState ������ ��ȯ�Ѵ�.
	 *	EState::E_TIMEOUT�� ��� ���ڷ� ������ timeoutInterval_msec �ð��� �ʰ��Ͽ��� ��� ��ȯ�ȴ�.
	 */
	EState send(const unsigned char *data, const size_t dataSize, const unsigned long timeoutInterval_msec = 100);
	/**
	 *	@brief  ����� serial ��� channel�� ���� data�� ����.
	 *	@param[out]  recvBuf  ���ŵ� data�� ������ receive data buffer.
	 *	@param[in]  timeoutInterval_msec  data ������ ���� ����� time-out interval. [msec] ������ ����.
	 *	@param[in]  bufferLen  asynchronous I/O�� ���� data�� ������ �� ���� buffer�� ũ�⸦ ����. 0���� �����ȴٸ� �ش� buffer�� ũ��� 8192 ũ��� �Ҵ�.
	 *	@return  ��� channel�� ��� ���¸� ��ȯ.
	 *
	 *	serial ����� ���� ���ŵǴ� data�� ���ڷ� ������ ByteBuffer class�� ��ü�� �����Ѵ�.
	 *
	 *	asynchronous I/O�� ���� data�� �����ϸ� �� �������� �߻��ϴ� channel ���¸� EState ������ ��ȯ�Ѵ�.
	 *	EState::E_TIMEOUT�� ��� ���ڷ� ������ timeoutInterval_msec �ð��� �ʰ��Ͽ��� ��� ��ȯ�ȴ�.
	 */
	EState receive(GuardedByteBuffer &recvBuf, const unsigned long timeoutInterval_msec = 100, const size_t bufferLen = 0);

	/**
	 *	@brief  ���� ���� I/O �۾��� ��ҽ�Ŵ.
	 *	@return  I/O �۾��� ���������� ����Ǿ��ٸ� true�� ��ȯ.
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
