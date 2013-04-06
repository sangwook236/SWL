#if !defined(__SWL_TCP_SOCKET_SERVER_TEST__ECHO_TCP_SOCKET_CONNECTION__H_)
#define __SWL_TCP_SOCKET_SERVER_TEST__ECHO_TCP_SOCKET_CONNECTION__H_ 1


#include "swl/util/TcpSocketConnection.h"


namespace swl {

//-----------------------------------------------------------------------------------
//

/**
 *	@brief  asynchronous I/O mode로 동작하는 TCP echo socket server의 channel 연결을 처리하는 connection utility class.
 *
 *	server에서 client의 접속 요청을 accept한 후, TCP socket connection 객체에 접속된 client의 접속 관리 및 처리를 수행하게 한다.
 *	개략적인 과정은 아래와 같다.
 *		-# [client] 접속 요청
 *		-# [server] client로부터 접속 요청 승인
 *		-# [server] connection 객체를 client와 연결
 *		-# [connection] socket 통신을 이용해 client와 message 송수신 수행
 *		-# [server] 다른 client의 접속 요청 대기
 *
 *	TCP socket 통신은 asynchronous I/O를 이용하여 수행한다.
 */
class EchoTcpSocketConnection: public TcpSocketConnection
{
public:
	typedef TcpSocketConnection base_type;
	typedef boost::shared_ptr<EchoTcpSocketConnection> pointer;

private:
	/**
	 *	@brief  [ctor] private constructor.
	 *	@param[in]  ioService  TCP socket 통신을 위한 Boost.ASIO의 I/O service 객체.
	 *
	 *	TCP socket connection 객체의 초기화를 수행한다.
	 */
	EchoTcpSocketConnection(boost::asio::io_service &ioService);
public:
	/**
	 *	@brief  [dtor] virtual default destructor.
	 *
	 *	TCP socket 통신 connection을 종료하기 위한 작업을 수행한다.
	 */
	virtual ~EchoTcpSocketConnection()  {}

public:
	/**
	 *	@brief  [ctor] TCP socket connection 객체의 생성을 위한 factory 함수.
	 *	@param[in]  ioService  TCP socket 통신을 위한 Boost.ASIO의 I/O service 객체.
	 *
	 *	TCP socket connection 객체의 instance를 생성한다.
	 */
	static pointer create(boost::asio::io_service &ioService);

private:
	/**
	 *	@brief  송신 buffer에 저장된 message의 전송을 시작.
	 *
	 *	송신 buffer에 저장되어 있는 message를 asynchronous I/O를 통해 송신한다.
	 */
	/*virtual*/ void doStartOperation();
	/**
	 *	@brief  송신 요청된 message의 전송이 완료된 경우 호출되는 completion routine.
	 *	@param[in]  ec  message를 전송하는 과정에서 발생한 오류의 error code.
	 *	@throw  LogException  TCP socket의 close 과정에서 error가 발생.
	 *
	 *	asynchronous I/O를 이용하여 송신 요청된 message의 전송이 완료되었을 때 system에 의해 호출되는 completion routine이다.
	 *	startSending() 함수 내에서 asynchronous 송신 요청을 하면서 해당 함수를 completion routine으로 지정해 주어야 한다.
	 */
	/*virtual*/ void doCompleteSending(boost::system::error_code ec);
	/**
	 *	@brief  TCP socket 통신 channel을 통해 수신된 message가 있는 경우 호출되는 completion routine.
	 *	@param[in]  ec  message를 수신하는 과정에서 발생한 오류의 error code.
	 *	@param[in]  bytesTransferred  수신된 message의 길이.
	 *	@throw  LogException  TCP socket의 close 과정에서 error가 발생.
	 *
	 *	asynchronous I/O를 통해 message의 수신되는 경우 system에 의해 호출되는 completion routine이다.
	 *	startReceiving() 함수 내에서 asynchronous 수신 요청을 하면서 해당 함수를 completion routine으로 지정해 주어야 한다.
	 */
	/*virtual*/ void doCompleteReceiving(boost::system::error_code ec, std::size_t bytesTransferred);
};

}  // namespace swl


#endif  // __SWL_TCP_SOCKET_SERVER_TEST__ECHO_TCP_SOCKET_CONNECTION__H_
