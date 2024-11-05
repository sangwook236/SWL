#if !defined(__SWL_UTIL__TCP_SOCKET_SERVER__H_)
#define __SWL_UTIL__TCP_SOCKET_SERVER__H_ 1


#include <boost/asio.hpp>
#include <boost/bind.hpp>


namespace swl {

//-----------------------------------------------------------------------------------
//

/**
 *	@brief  TCP socket ����� �����ϴ� server class.
 *
 *	���������� asynchronous I/O�� ����ϰ� �����Ƿ� �� class�� ����ϴ� S/W�� �δ��� ���� �ִ� ������ �ִ�.
 *
 *	��ü�� �����Ǹ� ������ port�� ��û�Ǵ� client�� ������ ó���ϰ� Connection ��ü�� ������ client�� ����� �����ϰ� �Ѵ�.
 *	�������� ������ �Ʒ��� ����.
 *		-# [client] ���� ��û
 *		-# [server] client�κ��� ���� ��û ����
 *		-# [server] connection ��ü�� client�� ����
 *		-# [connection] client���� ����� ����
 *		-# [server] �ٸ� client�� ���� ��û ���
 *
 *	�� �� client���� �������� ����� template parameter�� �Ѱ��� Connection ��ü�� ���� �����ϰ� �Ǵµ� Connection ��ü�� �Ʒ��� ��Ҹ� ������ �־�� �Ѵ�.
 *		- type definition
 *			- pointer;
 *		- interface
 *			- static pointer create(boost::asio::io_service &ioService);
 *			- boost::asio::ip::tcp::socket & getSocket(); and/or const boost::asio::ip::tcp::socket & getSocket() const;
 *			- void start();
 */
template<typename Connection>
class TcpSocketServer
{
public:
	//typedef TcpSocketServer base_type;
	typedef Connection connection_type;

public:
	/**
	 *	@brief  [ctor] constructor.
	 *	@param[in]  ioService  TCP socket ����� ���� Boost.ASIO�� I/O service ��ü.
	 *	@param[in]  portNum  TCP socket ����� ���� server�� open�ϴ� port ��ȣ.
	 *
	 *	TCP socket connection ��ü�� �ʱ�ȭ�� �����ϰ� ������ port�� ���Ǵ� client�� ���� ��û�� ��ٸ���.
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
