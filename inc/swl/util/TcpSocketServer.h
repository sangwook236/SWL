#if !defined(__SWL_UTIL__TCP_SOCKET_SERVER__H_)
#define __SWL_UTIL__TCP_SOCKET_SERVER__H_ 1


#include <boost/asio.hpp>
#include <boost/bind.hpp>


namespace swl {

//-----------------------------------------------------------------------------------
//

/**
 *	@brief  TCP socket 통신을 수행하는 server class.
 *
 *	내부적으로 asynchronous I/O를 사용하고 있으므로 이 class를 사용하는 S/W에 부담을 적게 주는 장점이 있다.
 *
 *	객체가 생성되면 지정된 port로 요청되는 client의 접속을 처리하고 Connection 객체를 생성해 client의 통신을 수행하게 한다.
 *	개략적인 과정은 아래와 같다.
 *		-# [client] 접속 요청
 *		-# [server] client로부터 접속 요청 승인
 *		-# [server] connection 객체를 client와 연결
 *		-# [connection] client와의 통신을 시작
 *		-# [server] 다른 client의 접속 요청 대기
 */
template<typename Connection>
class TcpSocketServer
{
public:
	//typedef TcpSocketServer base_type;
	typedef Connection connection_type;

public:
	/**
	 *	@brief  [ctor] contructor.
	 *	@param[in]  ioService  TCP socket 통신을 위한 Boost.ASIO의 I/O service 객체.
	 *	@param[in]  portNum  TCP socket 통신을 위한 server가 open할 port 번호.
	 *
	 *	TCP socket connection 객체의 초기화를 수행하고 지정된 port로 들어되는 client의 접속 요청을 기다린다.
	 */
	TcpSocketServer(boost::asio::io_service &ioService, const unsigned short portNum)
	: acceptor_(ioService, boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), portNum))
	{
		startAccepting();
	}

private:
	void startAccepting()
	{
		typename connection_type::pointer newConnection = connection_type::create(acceptor_.get_io_service());

		acceptor_.async_accept(
			newConnection->getSocket(),
			boost::bind(&TcpSocketServer::handleAccepting, this, newConnection, boost::asio::placeholders::error)
		);
	}

	void handleAccepting(typename connection_type::pointer newConnection, const boost::system::error_code &ec)
	{
		if (!ec)
		{
			newConnection->start();
			startAccepting();
		}
	}

private:
	boost::asio::ip::tcp::acceptor acceptor_;
};

}  // namespace swl


#endif  // __SWL_UTIL__TCP_SOCKET_SERVER__H_
