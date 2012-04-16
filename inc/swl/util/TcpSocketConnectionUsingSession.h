#if !defined(__SWL_UTIL__TCP_SOCKET_CONNECTION_USING_SESSION__H_)
#define __SWL_UTIL__TCP_SOCKET_CONNECTION_USING_SESSION__H_ 1


#include <boost/asio.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/bind.hpp>


namespace swl {

//-----------------------------------------------------------------------------------
//

/**
 *	@brief  3rd party library(session)를 사용하고 half/full duplex mode로 동작하는 TCP socket server의 channel 연결을 처리하는 connection utility class.
 *
 *	server에서 client의 접속 요청을 accept한 후, TCP socket connection 객체에 접속된 client의 접속 관리 및 처리를 수행하게 한다.
 *	개략적인 과정은 아래와 같다.
 *		-# [client] 접속 요청
 *		-# [server] client로부터 접속 요청 승인
 *		-# [server] connection 객체를 client와 연결
 *		-# [connection] client와의 통신을 위해 sesseion 객체를 시작
 *		-# [session] socket 통신을 이용해 client와 message 송수신 수행 (half/full duplex mode)
 *		-# [server] 다른 client의 접속 요청 대기
 *
 *	이 때 client와의 통신은 template parater로 넘겨지 Session 객체에 위임되며 Session 객체는 아래의 요소를 가지고 있어야 한다.
 *		- interface
 *			- bool isReadyToSend();
 *			- void send(boost::system::error_code &ec);
 *			- bool isReadyToSend();
 *			- void receive(boost::system::error_code &ec);
 *
 *	TCP socket 통신은 asynchronous I/O를 이용하여 수행한다.
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
	 *	@param[in]  ioService  TCP socket 통신을 위한 Boost.ASIO의 I/O service 객체.
	 *
	 *	TCP socket connection 객체의 초기화를 수행한다.
	 */
	TcpSocketConnectionUsingSession(boost::asio::io_service &ioService)
	: socket_(ioService), session_(socket_),
	  isReceiving_(false), isSending_(false)
	{}

public:
	/**
	 *	@brief  [ctor] TCP socket connection 객체의 생성을 위한 factory 함수.
	 *	@param[in]  ioService  TCP socket 통신을 위한 Boost.ASIO의 I/O service 객체.
	 *
	 *	TCP socket connection 객체의 instance를 생성한다.
	 */
	static pointer create(boost::asio::io_service &ioService)
	{  return pointer(new TcpSocketConnectionUsingSession(ioService));  }

	/**
	 *	@brief  TCP socket 통신을 수행하는 socket 객체를 반환.
	 *	@return  실제 TCP socket 통신를 담당하는 socket 객체.
	 *
	 *	실제로 TCP socket 통신을 수행하게 되는 socket 객체의 reference를 반환한다.
	 */
	boost::asio::ip::tcp::socket & getSocket()  {  return socket_;  }
	/**
	 *	@brief  TCP socket 통신을 수행하는 socket 객체를 반환.
	 *	@return  실제 TCP socket 통신를 담당하는 socket 객체.
	 *
	 *	실제로 TCP socket 통신을 수행하게 되는 socket 객체의 const reference를 반환한다.
	 */
	const boost::asio::ip::tcp::socket & getSocket() const  {  return socket_;  }

	/**
	 *	@brief  client와 TCP socket 통신을 시작.
	 *
	 *	TCP socket server를 통해 client와의 접속이 이루어진 후 client와 message의 송수신을 시작한다.
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
			boost::shared_ptr<TcpSocketConnectionUsingSession<Session> > pp = shared_from_this<TcpSocketConnectionUsingSession<Session> >();
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
				boost::bind(&TcpSocketConnectionUsingSession::completeSending, shared_from_this(), boost::asio::placeholders::error)
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
