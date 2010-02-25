#if !defined(__SWL_TCP_SOCKET_SERVER_TEST__ECHO_TCP_SOCKET_SESSION__H_)
#define __SWL_TCP_SOCKET_SERVER_TEST__ECHO_TCP_SOCKET_SESSION__H_ 1


#include "swl/util/TcpSocketSession.h"


namespace swl {

//-----------------------------------------------------------------------------------
//

/**
 *	@brief  TCP socket 통신을 수행하는 server session class.
 *
 *	TCP socket server의 connection 객체 (참고: TcpSocketConnectionUsingSession) 에서 사용하기 위해 설계된 session class이다.
 */
class EchoTcpSocketSession: public TcpSocketSession
{
public:
	typedef TcpSocketSession base_type;

public:
	/**
	 *	@brief  [ctor] contructor.
	 *	@param[in]  socket  TCP socket 통신을 위한 Boost.ASIO의 socket 객체.
	 *
	 *	TCP socket 통신 session을 위해 필요한 설정들을 초기화한다.
	 */
	EchoTcpSocketSession(boost::asio::ip::tcp::socket &socket);
	/**
	 *	@brief  [dtor] virtual default destructor.
	 *
	 *	TCP socket 통신 session을 종료하기 위해 필요한 절차를 수행한다.
	 */
	virtual ~EchoTcpSocketSession();

public:
	/**
	 *	@brief  send buffer에 있는 message를 연결된 TCP socket 통신 channel을 통해 전송.
	 *	@param[out]  ec  message 전송 과정에서 발생한 오류의 error code 객체.
	 *
	 *	요청된 message를 TCP socket 통신을 통해 전송한다.
	 *
	 *	TCP socket 통신을 통해 message를 전송하는 동안 발생한 오류의 error code를 인자로 넘어온다.
	 */
	/*virtual*/ void send(boost::system::error_code &ec);

	/**
	 *	@brief  TCP socket 통신 channel을 통해 수신된 message를 receive buffer에 저장.
	 *	@param[out]  ec  message 수신 과정에서 발생한 오류의 error code 객체.
	 *
	 *	TCP socket 통신을 통해 수신된 message를 receive buffer에 저장한다.
	 *
	 *	TCP socket 통신을 통해 message를 수신하는 동안 발생한 오류의 error code를 인자로 넘어온다.
	 */
	/*virtual*/ void receive(boost::system::error_code &ec);
};

}  // namespace swl


#endif  // __SWL_TCP_SOCKET_SERVER_TEST__ECHO_TCP_SOCKET_SESSION__H_
