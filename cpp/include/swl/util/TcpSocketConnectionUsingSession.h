#if !defined(__SWL_UTIL__TCP_SOCKET_CONNECTION_USING_SESSION__H_)
#define __SWL_UTIL__TCP_SOCKET_CONNECTION_USING_SESSION__H_ 1


#include <boost/asio.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/bind.hpp>


namespace swl {

//-----------------------------------------------------------------------------------
//

/**
 *	@brief  3rd party library(session)�� ����ϰ� half/full duplex mode�� �����ϴ� TCP socket server�� channel ������ ó���ϴ� connection utility class.
 *
 *	server���� client�� ���� ��û�� accept�� ��, TCP socket connection ��ü�� ���ӵ� client�� ���� ���� �� ó���� �����ϰ� �Ѵ�.
 *	�������� ������ �Ʒ��� ����.
 *		-# [client] ���� ��û
 *		-# [server] client�κ��� ���� ��û ����
 *		-# [server] connection ��ü�� client�� ����
 *		-# [connection] client���� ����� ���� sesseion ��ü�� ����
 *		-# [session] socket ����� �̿��� client�� message �ۼ��� ���� (half/full duplex mode)
 *		-# [server] �ٸ� client�� ���� ��û ���
 *
 *	�� �� client���� ����� template parater�� �Ѱ��� Session ��ü�� ���ӵǸ� Session ��ü�� �Ʒ��� ��Ҹ� ������ �־�� �Ѵ�.
 *		- interface
 *			- bool isReadyToSend();
 *			- void send(boost::system::error_code &ec);
 *			- bool isReadyToSend();
 *			- void receive(boost::system::error_code &ec);
 *
 *	TCP socket ����� asynchronous I/O�� �̿��Ͽ� �����Ѵ�.
 */
template<typename Session>
class TcpSocketConnectionUsingSession: public boost::enable_shared_from_this<TcpSocketConnectionUsingSession<Session> >
{
public:
	//typedef TcpSocketConnectionUsingSession base_type;
	typedef Session session_type;
	typedef boost::shared_ptr<TcpSocketConnectionUsingSession> pointer;

private:
	/**
	 *	@brief  [ctor] private constructor.
	 *	@param[in]  ioService  TCP socket ����� ���� Boost.ASIO�� I/O service ��ü.
	 *
	 *	TCP socket connection ��ü�� �ʱ�ȭ�� �����Ѵ�.
	 */
	TcpSocketConnectionUsingSession(boost::asio::io_service &ioService)
	: socket_(ioService), session_(socket_),
	  isReceiving_(false), isSending_(false)
	{}

public:
	/**
	 *	@brief  [ctor] TCP socket connection ��ü�� ������ ���� factory �Լ�.
	 *	@param[in]  ioService  TCP socket ����� ���� Boost.ASIO�� I/O service ��ü.
	 *
	 *	TCP socket connection ��ü�� instance�� �����Ѵ�.
	 */
	static pointer create(boost::asio::io_service &ioService)
	{  return pointer(new TcpSocketConnectionUsingSession(ioService));  }

	/**
	 *	@brief  TCP socket ����� �����ϴ� socket ��ü�� ��ȯ.
	 *	@return  ���� TCP socket ��Ÿ� ����ϴ� socket ��ü.
	 *
	 *	������ TCP socket ����� �����ϰ� �Ǵ� socket ��ü�� reference�� ��ȯ�Ѵ�.
	 */
	boost::asio::ip::tcp::socket & getSocket()  {  return socket_;  }
	/**
	 *	@brief  TCP socket ����� �����ϴ� socket ��ü�� ��ȯ.
	 *	@return  ���� TCP socket ��Ÿ� ����ϴ� socket ��ü.
	 *
	 *	������ TCP socket ����� �����ϰ� �Ǵ� socket ��ü�� const reference�� ��ȯ�Ѵ�.
	 */
	const boost::asio::ip::tcp::socket & getSocket() const  {  return socket_;  }

	/**
	 *	@brief  client�� TCP socket ����� ����.
	 *
	 *	TCP socket server�� ���� client���� ������ �̷���� �� client�� message�� �ۼ����� �����Ѵ�.
	 */
	void start()
	{
		// put the socket into non-blocking mode.
		boost::asio::ip::tcp::socket::non_blocking_io non_blocking_io(true);
		socket_.io_control(non_blocking_io);

		startOperation();
	}

private:
	void startOperation()
	{
		// start a read operation if the third party library wants one.
		if (session_.isReadyToReceive() && !isReceiving_)
		{
			isReceiving_ = true;
#if defined(__GNUC__)
			boost::shared_ptr<TcpSocketConnectionUsingSession<session_type> > pp = this->shared_from_this();
#else
			boost::shared_ptr<TcpSocketConnectionUsingSession<session_type> > pp = shared_from_this();
#endif
			socket_.async_read_some(
				boost::asio::null_buffers(),
				boost::bind(&TcpSocketConnectionUsingSession::completeReceiving, pp, boost::asio::placeholders::error)
			);
		}

		// start a write operation if the third party library wants one.
		if (session_.isReadyToSend() && !isSending_)
		{
			isSending_ = true;
			socket_.async_write_some(
				boost::asio::null_buffers(),
#if defined(__GNUC__)
				boost::bind(&TcpSocketConnectionUsingSession::completeSending, this->shared_from_this(), boost::asio::placeholders::error)
#else
				boost::bind(&TcpSocketConnectionUsingSession::completeSending, shared_from_this(), boost::asio::placeholders::error)
#endif
			);
		}
	}

	void completeSending(boost::system::error_code ec)
	{
		isSending_ = false;

		// notify third party library that it can perform a write.
		if (!ec)
			session_.send(ec);

		// the third party library successfully performed a write on the socket.
		// start new read or write operations based on what it now wants.
		if (!ec || boost::asio::error::would_block == ec)
			startOperation();
		// otherwise, an error occurred. Closing the socket cancels any outstanding asynchronous read or write operations.
		// the TcpSocketConnectionUsingSession object will be destroyed automatically once those outstanding operations complete.
		else
			socket_.close();
	}

	void completeReceiving(boost::system::error_code ec)
	{
		isReceiving_ = false;

		// notify third party library that it can perform a read.
		if (!ec)
			session_.receive(ec);

		// the third party library successfully performed a read on the socket.
		// start new read or write operations based on what it now wants.
		if (!ec || boost::asio::error::would_block == ec)
			startOperation();
		// otherwise, an error occurred. Closing the socket cancels any outstanding asynchronous read or write operations.
		// the TcpSocketConnection_FullDuplex object will be destroyed automatically once those outstanding operations complete.
		else
			socket_.close();
	}

private:
	boost::asio::ip::tcp::socket socket_;

	session_type session_;

	bool isReceiving_;
	bool isSending_;
};

}  // namespace swl


#endif  // __SWL_UTIL__TCP_SOCKET_CONNECTION_USING_SESSION__H_
