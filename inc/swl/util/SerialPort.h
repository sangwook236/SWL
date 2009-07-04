#if !defined(__SWL_UTIL__SERIAL_PORT__H_)
#define __SWL_UTIL__SERIAL_PORT__H_ 1


#include "swl/util/ExportUtil.h"
#include "swl/util/GuardedBuffer.h"
#include <boost/asio.hpp>
#include <string>


namespace swl {

//-----------------------------------------------------------------------------------
//

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
 *	serial 통신을 통해 message를 전송할 경우에는 send() 함수를 해당 시점에 직접 호출하면 되고,
 *	수신의 경우에는 receive() 함수를 호출하면 된다.
 *
 *	serial 통신을 실제로 수행하기 위해서는 connect()를 함수를 호출하여 port를 연결한 후
 *	constructor의 인자로 넘겨준 I/O를 run시켜야 한다.
 */
class SWL_UTIL_API SerialPort
{
public:
	//typedef SerialPort base_type;

public:
	/**
	 *	@brief  [ctor] contructor.
	 *	@param[in]  ioService  serial 통신을 위한 Boost.ASIO의 I/O service 객체.
	 *
	 *	serial 통신을 위해 필요한 설정들을 초기화한다.
	 */
	SerialPort(boost::asio::io_service &ioService);
	/**
	 *	@brief  [dtor] default destructor.
	 *
	 *	serial 통신을 종료하기 위해 필요한 절차를 수행한다.
	 *	통신 port가 열려 있는 경우 disconnect() 함수를 호출하여 이를 닫는다.
	 */
	virtual ~SerialPort();

public:
	/**
	 *	@brief  지정된 COM port와 Baud rate로 serial 통신 channel을 연결.
	 *	@param[in]  portName  serial 통신을 위한 port 이름.
	 *	@param[in]  baudRate  통신을 위해 사용하고자 하는 속도.
	 *	@return  serial 통신 channel이 정상적으로 연결되었다면 true 반환.
	 *
	 *	인자로 넘겨진 port 이름과 Baud rate를 이용하여 통신 채널을 연결하고
	 *	asynchronous I/O 작업을 수행하기 위한 작업을 수행한다.
	 */
#if defined(_UNICODE) || defined(UNICODE)
	bool connect(const std::wstring &portName, const unsigned int baudRate);
#else
	bool connect(const std::string &portName, const unsigned int baudRate);
#endif
	/**
	 *	@brief  serial 통신을 위한 port를 닫음.
	 *	@throw  LogException  serial port의 close 과정에서 error가 발생.
	 *
	 *	serial 통신을 위해 연결하였던 channel을 끊고, 활용한 resource를 반환한다.
	 */
	void disconnect();

	/**
	 *	@brief  serial 통신을 위한 port가 연결되어 있는지 확인.
	 *	@return  serial 통신 channel이 연결되어 있다면 true 반환.
	 *
	 *	serial 통신 channel의 연결 상태를 반환한다.
	 */
	bool isConnected() const  {  return isActive_;  }

	/**
	 *	@brief  지정된 message를 연결된 serial 통신 channel을 통해 전송.
	 *	@param[in]  msg  전송할 message를 지정하는 pointer.
	 *	@param[in]  len  전송할 message 길이.
	 *
	 *	요청된 message를 serial 통신을 통해 전송한다.
	 *	asynchronous I/O를 통해 message를 전송한다.
	 */
	void send(const unsigned char *msg, const size_t len);
	/**
	 *	@brief  연결된 serial 통신 channel을 통해 message를 수신.
	 *	@param[out]  msg  수신된 message를 저장할 pointer.
	 *	@param[in]  len  asynchronous I/O를 통해 수신한 message를 저장할 buffer의 크기를 지정.
	 *	@return  실제로 수신된 message의 길이를 반환. 인자로 지정된 len보다 작거나 같음.
	 *
	 *	serial 통신을 통해 수신되는 message를 인자로 지정된 pointer의 객체에 저장한다.
	 *	asynchronous I/O를 통해 message를 수신한다.
	 */
	size_t receive(unsigned char *msg, const size_t len);

	/**
	 *	@brief  수행 중인 I/O 작업을 취소.
	 *	@throw  LogException  송수신 operation을 취소하는 과정에서 error가 발생.
	 *
	 *	asynchronous I/O를 통해 진행 중인 송수신 operation을 취소한다.
	 */
	void cancelIo();

	/**
	 *	@brief  serial 통신의 송신 buffer를 비움.
	 *
	 *	전송되지 않은 송신 buffer의 모든 message를 제거한다.
	 *	하지만 송신 message의 내용이 무엇인지 알 수 없으므로 예기치 않은 error를 발생시킬 수 있다.
	 */
	void clearSendBuffer();
	/**
	 *	@brief  serial 통신의 수신 buffer를 비움.
	 *
	 *	serial 통신 channel로 수신된 수신 buffer의 모든 message를 제거한다.
	 *	하지만 수신 message의 내용이 무엇인지 알 수 없으므로 예기치 않은 error를 발생시킬 수 있다.
	 */
	void clearReceiveBuffer();

	/**
	 *	@brief  serial 통신의 송신 buffer가 비어 있는지를 확인.
	 *	@return  송신 buffer가 비어 있다면 true를 반환.
	 *
	 *	serial 통신을 통해 전송할 message의 송신 buffer가 비어 있는지 여부를 반환한다.
	 */
	bool isSendBufferEmpty() const;
	/**
	 *	@brief  serial 통신의 수신 buffer가 비어 있는지를 확인.
	 *	@return  수신 buffer가 비어 있다면 true를 반환.
	 *
	 *	serial 통신을 통해 수신된 message의 수신 buffer가 비어 있는지 여부를 반환한다.
	 */
	bool isReceiveBufferEmpty() const;

	/**
	 *	@brief  serial 통신을 통해 송신할 message의 길이를 반환.
	 *	@return  송신 message의 길이를 반환.
	 *
	 *	serial 통신을 통해 전송할 message를 저장하고 있는 송신 buffer의 길이를 반환한다.
	 */
	size_t getSendBufferSize() const;
	/**
	 *	@brief  serial 통신을 통해 수신된 message의 길이를 반환.
	 *	@return  수신된 message의 길이를 반환.
	 *
	 *	serial 통신을 통해 수신된 message를 저장하고 있는 수신 buffer의 길이를 반환한다.
	 */
	size_t getReceiveBufferSize() const;

protected:
	/**
	 *	@brief  송신 buffer에 저장된 message의 전송을 시작.
	 *
	 *	송신 buffer에 저장되어 있는 message를 asynchronous I/O를 통해 송신한다.
	 */
	virtual void startSending();
	/**
	 *	@brief  송신 요청된 message의 전송이 완료된 경우 호출되는 completion routine.
	 *	@param[in]  ec  message를 전송하는 과정에서 발생한 오류의 error code.
	 *	@throw  LogException  serial port의 close 과정에서 error가 발생.
	 *
	 *	asynchronous I/O를 이용하여 송신 요청된 message의 전송이 완료되었을 때 system에 의해 호출되는 completion routine이다.
	 *	startSending() 함수 내에서 asynchronous 송신 요청을 하면서 해당 함수를 completion routine으로 지정해 주어야 한다.
	 */
	virtual void completeSending(const boost::system::error_code &ec);
	/**
	 *	@brief  serial 통신 channel을 통해 들어오는 message를 receive buffer로 수신 시작.
	 *
	 *	serial 통신을 수신되는 message를 asynchronous I/O를 이용하여 수신하기 시작한다.
	 */
	virtual void startReceiving();
	/**
	 *	@brief  serial 통신 channel을 통해 수신된 message가 있는 경우 호출되는 completion routine.
	 *	@param[in]  ec  message를 수신하는 과정에서 발생한 오류의 error code.
	 *	@param[in]  bytesTransferred  수신된 message의 길이.
	 *	@throw  LogException  serial port의 close 과정에서 error가 발생.
	 *
	 *	asynchronous I/O를 통해 message의 수신되는 경우 system에 의해 호출되는 completion routine이다.
	 *	startReceiving() 함수 내에서 asynchronous 수신 요청을 하면서 해당 함수를 completion routine으로 지정해 주어야 한다.
	 */
	virtual void completeReceiving(const boost::system::error_code &ec, size_t bytesTransferred);

private:
	void doSendOperation(const unsigned char *msg, const size_t len);
	void doCloseOperation(const boost::system::error_code &ec);
	void doCancelOperation(const boost::system::error_code &ec);
	
private:
	static const size_t MAX_SEND_LENGTH_ = 512;
	static const size_t MAX_RECEIVE_LENGTH_ = 512;

	boost::asio::io_service &ioService_;
	boost::asio::serial_port port_;

	bool isActive_;

	GuardedByteBuffer sendBuffer_;
	GuardedByteBuffer receiveBuffer_;
	GuardedByteBuffer::value_type sendMsg_[MAX_SEND_LENGTH_];
	GuardedByteBuffer::value_type receiveMsg_[MAX_RECEIVE_LENGTH_];
	size_t sentMsgLength_;
};

}  // namespace swl


#endif  // __SWL_UTIL__SERIAL_PORT__H_
