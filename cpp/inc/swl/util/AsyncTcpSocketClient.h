#if !defined(__SWL_UTIL__ASYNC_TCP_SOCKET_CLIENT__H_)
#define __SWL_UTIL__ASYNC_TCP_SOCKET_CLIENT__H_ 1


#include "swl/util/ExportUtil.h"
#include "swl/util/GuardedBuffer.h"
#include <boost/asio.hpp>
#include <boost/array.hpp>
#include <string>


namespace swl {

//-----------------------------------------------------------------------------------
//

/**
 *	@brief  비동기적으로 TCP socket 통신을 수행하는 client class.
 *
 *	TCP socket 통신을 통해 message를 송수신하기 위해 send() 함수와 receive() 함수를 호출하면 된다.
 *	TCP socket 통신을 수행하는 개략적인 절차는 아래와 같다.
 *		- AsyncTcpSocketClient 객체 생성
 *		- doStartConnecting() 함수를 이용하여 TCP server와 연결
 *		- send() and/or receive() 함수를 이용해 message 송수신
 *		- 작업이 끝났다면, disconnect() 함수를 호출하여 연결 해제
 *		- AsyncTcpSocketClient 객체 소멸
 *
 *	asynchronous I/O를 사용하여 송수신을 수행한다.
 */
class SWL_UTIL_API AsyncTcpSocketClient
{
public:
	//typedef AsyncTcpSocketClient base_type;

public:
	/**
	 *	@brief  [ctor] constructor.
	 *	@param[in]  ioService  TCP socket 통신을 위한 Boost.ASIO의 I/O service 객체.
	 *	@param[in]  hostName  TCP socket server의 host 이름.
	 *	@param[in]  serviceName  TCP socket server의 service 이름.
	 *
	 *	TCP socket 통신을 위해 필요한 설정들을 초기화하고,
	 *	인자로 넘겨진 host 이름과 service 이름을 이용하여 TCP socket channel을 연결한다.
	 *
	 *	host 이름은 IP address이나 domain 이름으로 지정할 수 있다.
	 *		- "abc.com"
	 *		- "100.110.120.130"
	 *	service 이름은 이나 port 번호로 지정할 수 있다.
	 *		- "http" or "daytime"
	 *		- "80"
	 */
#if defined(_UNICODE) || defined(UNICODE)
	AsyncTcpSocketClient(boost::asio::io_service &ioService, const std::wstring &hostName, const std::wstring &serviceName);
#else
	AsyncTcpSocketClient(boost::asio::io_service &ioService, const std::string &hostName, const std::string &serviceName);
#endif
	/**
	 *	@brief  [dtor] virtual default destructor.
	 *
	 *	TCP socket 통신을 종료하기 위한 절차를 수행한다.
	 *	통신 channel이 열려 있는 경우 disconnect() 함수를 호출하여 이를 닫는다.
	 */
	virtual ~AsyncTcpSocketClient();

public:
	/**
	 *	@brief  TCP socket 통신 channel의 연결을 해제.
	 *	@throw  LogException  TCP socket의 close 과정에서 error가 발생.
	 *
	 *	TCP socket 통신 channel의 연결을 끊고, 사용 중인 resource를 반환한다.
	 */
	void disconnect();

	/**
	 *	@brief  TCP socket 통신이 연결 상태에 있는지 확인.
	 *	@return  TCP socket 통신 channel이 연결 상태이면 true 반환.
	 *
	 *	TCP socket 통신 channel의 연결 상태를 반환한다.
	 */
	bool isConnected() const  {  return isActive_;  }

	/**
	 *	@brief  지정된 message를 연결된 TCP socket 통신 channel을 통해 전송.
	 *	@param[in]  msg  전송할 message를 지정하는 pointer.
	 *	@param[in]  len  전송할 message 길이.
	 *
	 *	요청된 message를 TCP socket 통신을 통해 전송한다.
	 *	asynchronous I/O를 통해 message를 전송한다.
	 */
	void send(const unsigned char *msg, const std::size_t len);
	/**
	 *	@brief  TCP socket 통신 channel을 통해 message를 수신.
	 *	@param[out]  msg  수신된 message를 저장할 pointer.
	 *	@param[in]  len  asynchronous I/O를 통해 수신한 message를 저장할 buffer의 크기를 지정.
	 *	@return  실제로 수신된 message의 길이를 반환. 인자로 지정된 len보다 작거나 같음.
	 *
	 *	TCP socket 통신을 통해 수신되는 message를 인자로 지정된 pointer의 객체에 저장한다.
	 *	asynchronous I/O를 통해 message를 수신한다.
	 */
	std::size_t receive(unsigned char *msg, const std::size_t len);

	/**
	 *	@brief  수행 중인 I/O 작업을 취소.
	 *	@throw  LogException  송수신 operation을 취소하는 과정에서 error가 발생.
	 *
	 *	asynchronous I/O를 통해 진행 중인 송수신 operation을 취소한다.
	 */
	void cancelIo();

	/**
	 *	@brief  TCP socket 통신의 송신 buffer를 비움.
	 *
	 *	전송되지 않은 송신 buffer의 모든 message를 제거한다.
	 *	하지만 송신 message의 내용이 무엇인지 알 수 없으므로 예기치 않은 error를 발생시킬 수 있다.
	 */
	void clearSendBuffer();
	/**
	 *	@brief  TCP socket 통신의 수신 buffer를 비움.
	 *
	 *	TCP socket 통신 channel로 수신된 수신 buffer의 모든 message를 제거한다.
	 *	하지만 수신 message의 내용이 무엇인지 알 수 없으므로 예기치 않은 error를 발생시킬 수 있다.
	 */
	void clearReceiveBuffer();

	/**
	 *	@brief  TCP socket 통신의 송신 buffer가 비어 있는지를 확인.
	 *	@return  송신 buffer가 비어 있다면 true를 반환.
	 *
	 *	TCP socket 통신을 통해 전송할 message의 송신 buffer가 비어 있는지 여부를 반환한다.
	 */
	bool isSendBufferEmpty() const;
	/**
	 *	@brief  TCP socket 통신의 수신 buffer가 비어 있는지를 확인.
	 *	@return  수신 buffer가 비어 있다면 true를 반환.
	 *
	 *	TCP socket 통신을 통해 수신된 message의 수신 buffer가 비어 있는지 여부를 반환한다.
	 */
	bool isReceiveBufferEmpty() const;

	/**
	 *	@brief  TCP socket 통신을 통해 송신할 message의 길이를 반환.
	 *	@return  송신 message의 길이를 반환.
	 *
	 *	TCP socket 통신을 통해 전송할 message를 저장하고 있는 송신 buffer의 길이를 반환한다.
	 */
	std::size_t getSendBufferSize() const;
	/**
	 *	@brief  TCP socket 통신을 통해 수신된 message의 길이를 반환.
	 *	@return  수신된 message의 길이를 반환.
	 *
	 *	TCP socket 통신을 통해 수신된 message를 저장하고 있는 수신 buffer의 길이를 반환한다.
	 */
	std::size_t getReceiveBufferSize() const;

protected:
	/**
	 *	@brief  asynchronous mode로 지정된 host 이름과 service 이름을 이용해 TCP socket server와 channel을 연결.
	 *	@param[in]  endpoint_iterator  TCP socket 통신을 통해 연결할 server의 end point.
	 *
	 *	인자로 넘겨진 host 이름과 service 이름을 이용하여 TCP socket channel을 연결한다.
	 */
	virtual void doStartConnecting(boost::asio::ip::tcp::resolver::iterator endpoint_iterator);
	/**
	 *	@brief  asynchronous I/O를 통해 요청된 server 접속 요청이 완료된 경우 호출되는 completion routine.
	 *	@param[in]  ec  server 접속 과정에서 발생한 오류의 error code.
	 *	@param[in]  endpoint_iterator  TCP socket 통신을 통해 연결할 server의 end point.
	 *
	 *	asynchronous I/O를 이용하여 요청된 접속 시도가 완료되었을 때 system에 의해 호출되는 completion routine이다.
	 *	접속이 정상적으로 완료되었다면 isConnected()가 true를 반환하게 된다.
	 */
	virtual void doCompleteConnecting(const boost::system::error_code &ec, boost::asio::ip::tcp::resolver::iterator endpoint_iterator);
	/**
	 *	@brief  송신 buffer에 저장된 message의 전송을 시작.
	 *
	 *	송신 buffer에 저장되어 있는 message를 asynchronous I/O를 통해 송신한다.
	 */
	virtual void doStartSending();
	/**
	 *	@brief  송신 요청된 message의 전송이 완료된 경우 호출되는 completion routine.
	 *	@param[in]  ec  message를 전송하는 과정에서 발생한 오류의 error code.
	 *	@throw  LogException  TCP socket channel의 close 과정에서 error가 발생.
	 *
	 *	asynchronous I/O를 이용하여 송신 요청된 message의 전송이 완료되었을 때 system에 의해 호출되는 completion routine이다.
	 *	doStartSending() 함수 내에서 asynchronous 송신 요청을 하면서 해당 함수를 completion routine으로 지정해 주어야 한다.
	 */
	virtual void doCompleteSending(const boost::system::error_code &ec);
	/**
	 *	@brief  TCP socket 통신 channel을 통해 들어오는 message를 receive buffer로 수신 시작.
	 *
	 *	TCP socket 통신을 수신되는 message를 asynchronous I/O를 이용하여 수신하기 시작한다.
	 */
	virtual void doStartReceiving();
	/**
	 *	@brief  TCP socket 통신 channel을 통해 수신된 message가 있는 경우 호출되는 completion routine.
	 *	@param[in]  ec  message를 수신하는 과정에서 발생한 오류의 error code.
	 *	@param[in]  bytesTransferred  수신된 message의 길이.
	 *	@throw  LogException  TCP socket channel의 close 과정에서 error가 발생.
	 *
	 *	asynchronous I/O를 통해 message의 수신되는 경우 system에 의해 호출되는 completion routine이다.
	 *	doStartReceiving() 함수 내에서 asynchronous 수신 요청을 하면서 해당 함수를 completion routine으로 지정해 주어야 한다.
	 */
	virtual void doCompleteReceiving(const boost::system::error_code &ec, std::size_t bytesTransferred);

private:
	void doSendOperation(const unsigned char *msg, const std::size_t len);
	void doCloseOperation(const boost::system::error_code &ec);
	void doCancelOperation(const boost::system::error_code &ec);

protected:
	/**
	 *	@brief  한 번의 송신 과정에서 보낼 수 있는 message의 최대 길이.
	 */
#if defined(__GNUC__)
	static const unsigned long MAX_SEND_LENGTH_ = 512;
#else
	static const std::size_t MAX_SEND_LENGTH_ = 512;
#endif
	/**
	 *	@brief  한 번의 수신 과정에서 받을 수 있는 message의 최대 길이.
	 */
#if defined(__GNUC__)
	static const unsigned long MAX_RECEIVE_LENGTH_ = 512;
#else
	static const std::size_t MAX_RECEIVE_LENGTH_ = 512;
#endif

	/**
	 *	@brief  TCP socket 통신을 실제적으로 수행하는 Boost.ASIO의 socket 객체.
	 */
	boost::asio::ip::tcp::socket socket_;

	/**
	 *	@brief  TCP socket 통신 channel이 연결되어 있고 정상 상태인지를 확인하는 flag 변수.
	 *
	 *	TCP socket 통신 channel이 연결되어 있고 정상 상태라면 true를, 그렇지 않다면 false를 표시한다.
	 */
	bool isActive_;

	/**
	 *	@brief  TCP socket 통신을 위한 send buffer.
	 *
	 *	GuardedByteBuffer의 객체로 multi-thread 환경에서도 안전하게 사용할 수 있다.
	 */
	GuardedByteBuffer sendBuffer_;
	/**
	 *	@brief  TCP socket 통신을 위한 send buffer.
	 *
	 *	GuardedByteBuffer의 객체로 multi-thread 환경에서도 안전하게 사용할 수 있다.
	 */
	GuardedByteBuffer receiveBuffer_;
	/**
	 *	@brief  한 번의 송신 과정에서 전송하게 될 message를 저장하는 buffer.
	 *
	 *	buffer의 길이는 MAX_SEND_LENGTH_이다.
	 */
	boost::array<GuardedByteBuffer::value_type, MAX_SEND_LENGTH_> sendMsg_;
	/**
	 *	@brief  한 번의 수신 과정에서 수신하게 될 message를 저장하는 buffer.
	 *
	 *	buffer의 길이는 MAX_RECEIVE_LENGTH_이다.
	 */
	boost::array<GuardedByteBuffer::value_type, MAX_RECEIVE_LENGTH_> receiveMsg_;
	/**
	 *	@brief  가장 최근 송신 과정에서 전송한 message의 길이.
	 */
	std::size_t sentMsgLength_;
};

}  // namespace swl


#endif  // __SWL_UTIL__ASYNC_TCP_SOCKET_CLIENT__H_
