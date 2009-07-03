#if !defined(__SWL_UTIL__TCP_SOCKET_CONNECTION__H_)
#define __SWL_UTIL__TCP_SOCKET_CONNECTION__H_ 1


#include <boost/asio.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/bind.hpp>


namespace swl {

//-----------------------------------------------------------------------------------
//

/**
 *	@brief  full duplex mode로 동작하는 TCP socket server의 channel 연결을 처리하는 connection utility class.
 *
 *	server에서 client의 접속 요청을 accept한 후, TCP socket connection 객체에 접속된 client의 접속 관리 및 처리를 수행하게 한다.
 *	개략적인 과정은 아래와 같다.
 *		-# [client] 접속 요청
 *		-# [server] client로부터 접속 요청 승인
 *		-# [server] connection 객체를 client와 연결
 *		-# [connection] client와의 통신을 위해 sesseion 객체를 시작 (full duplex mode)
 *		-# [session] socket 통신을 이용해 client와 message 송수신 수행
 *		-# [server] 다른 client의 접속 요청 대기
 *
 *	이 때 client와의 통신은 data member인 Session 객체에 의해 수행하게 되는데 Session 객체는 아래의 요소를 가지고 있어야 한다.
 *		- type definition
 *			- pointer;
 *		- interface
 *			- static pointer create(boost::asio::io_service &ioService);
 *			- boost::asio::ip::tcp::socket & getSocket(); and/or const boost::asio::ip::tcp::socket & getSocket() const;
 *			- void start();
 *
 *	TCP socket 통신은 asynchronous I/O를 이용하여 수행한다.
 */
template<typename Session>
class FullDuplexTcpSocketConnection: public boost::enable_shared_from_this<FullDuplexTcpSocketConnection>
{
public:
	//typedef FullDuplexTcpSocketConnection base_type;
	typedef Session session_type;
	typedef boost::shared_ptr<FullDuplexTcpSocketConnection> pointer;

private:
	/**
	 *	@brief  [ctor] private contructor.
	 *	@param[in]  ioService  TCP socket 통신을 위한 Boost.ASIO의 I/O service 객체.
	 *
	 *	TCP socket connection 객체의 초기화를 수행한다.
	 */
	FullDuplexTcpSocketConnection(boost::asio::io_service &ioService)
	: ioService_(ioService), socket_(ioService), session_(socket_)
	{}

public:
	/**
	 *	@brief  [ctor] TCP socket connection 객체의 생성을 위한 factory 함수.
	 *	@param[in]  ioService  TCP socket 통신을 위한 Boost.ASIO의 I/O service 객체.
	 *
	 *	TCP socket connection 객체의 instance를 생성한다.
	 */
	static pointer create(boost::asio::io_service &ioService)
	{  return pointer(new FullDuplexTcpSocketConnection(ioService));  }

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
	 *	@brief  TCP socket 통신 과정에서 실제 task를 수행하는 session 객체를 반환.
	 *	@return  실제 TCP socket 통신 과정의 task를 수행하는 session 객체.
	 *
	 *	TCP socket 통신 과정에서 message 송신 및 수신의 실제적인 task를 수행하는 session 객체의 reference를 반환한다.
	 */
	session_type & getSession()  {  return session_;  }
	/**
	 *	@brief  TCP socket 통신 과정에서 실제 task를 수행하는 session 객체를 반환.
	 *	@return  실제 TCP socket 통신 과정의 task를 수행하는 session 객체.
	 *
	 *	TCP socket 통신 과정에서 message 송신 및 수신의 실제적인 task를 수행하는 session 객체의 const reference를 반환한다.
	 */
	const session_type & getSession() const  {  return session_;  }

	/**
	 *	@brief  client와 TCP socket 통신을 시작.
	 *
	 *	TCP socket server를 통해 client와의 접속이 이루어진 후 client와 message의 송수신을 시작한다.
	 */
	void start()
	{
		startReceiving();
		// TODO [check] >>
		startSending();
	}

private:
	void startSending()
	{
		session_.startSending();
	}

	void startReceiving()
	{
		session_.startReceiving();
	}

private:
	boost::asio::io_service &ioService_;
	boost::asio::ip::tcp::socket socket_;

	session_type session_;
};

//-----------------------------------------------------------------------------------
//

/**
 *	@brief  half duplex mode로 동작하는 TCP socket server의 channel 연결을 처리하는 connection utility class.
 *
 *	server에서 client의 접속 요청을 accept한 후, TCP socket connection 객체에 접속된 client의 접속 관리 및 처리를 수행하게 한다.
 *	개략적인 과정은 아래와 같다.
 *		-# [client] 접속 요청
 *		-# [server] client로부터 접속 요청 승인
 *		-# [server] connection 객체를 client와 연결
 *		-# [connection] client와의 통신을 위해 sesseion 객체를 시작
 *		-# [session] socket 통신을 이용해 client와 message 송수신 수행 (half duplex mode)
 *		-# [server] 다른 client의 접속 요청 대기
 *
 *	이 때 client와의 통신은 data member인 Session 객체에 의해 수행하게 되는데 Session 객체는 아래의 요소를 가지고 있어야 한다.
 *		- type definition
 *			- pointer;
 *		- interface
 *			- static pointer create(boost::asio::io_service &ioService);
 *			- boost::asio::ip::tcp::socket & getSocket(); and/or const boost::asio::ip::tcp::socket & getSocket() const;
 *			- void start();
 *
 *	TCP socket 통신은 asynchronous I/O를 이용하여 수행한다.
 */
template<typename Session>
class HalfDuplexTcpSocketConnection: public boost::enable_shared_from_this<HalfDuplexTcpSocketConnection>
{
public:
	//typedef HalfDuplexTcpSocketConnection base_type;
	typedef Session session_type;
	typedef boost::shared_ptr<HalfDuplexTcpSocketConnection> pointer;

private:
	/**
	 *	@brief  [ctor] private contructor.
	 *	@param[in]  ioService  TCP socket 통신을 위한 Boost.ASIO의 I/O service 객체.
	 *
	 *	TCP socket connection 객체의 초기화를 수행한다.
	 */
		HalfDuplexTcpSocketConnection(boost::asio::io_service &ioService)
	: ioService_(ioService), socket_(ioService), session_(socket_),
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
	{  return pointer(new HalfDuplexTcpSocketConnection(ioService));  }

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
	 *	@brief  TCP socket 통신 과정에서 실제 task를 수행하는 session 객체를 반환.
	 *	@return  실제 TCP socket 통신 과정의 task를 수행하는 session 객체.
	 *
	 *	TCP socket 통신 과정에서 message 송신 및 수신의 실제적인 task를 수행하는 session 객체의 reference를 반환한다.
	 */
	session_type & getSession()  {  return session_;  }
	/**
	 *	@brief  TCP socket 통신 과정에서 실제 task를 수행하는 session 객체를 반환.
	 *	@return  실제 TCP socket 통신 과정의 task를 수행하는 session 객체.
	 *
	 *	TCP socket 통신 과정에서 message 송신 및 수신의 실제적인 task를 수행하는 session 객체의 const reference를 반환한다.
	 */
	const session_type & getSession() const  {  return session_;  }

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

		doStartOperation();
	}

private:
	void doStartOperation()
	{
		// start a read operation if the third party library wants one.
		if (session_.isReadyToReceive() && !isReceiving_)
		{
			isReceiving_ = true;
			socket_.async_read_some(
				boost::asio::null_buffers(),
				boost::bind(&HalfDuplexTcpSocketConnection::completeReceiving, shared_from_this(), boost::asio::placeholders::error)
			);
		}

		// start a write operation if the third party library wants one.
		if (session_.isReadyToSend() && !isSending_)
		{
			isSending_ = true;
			socket_.async_write_some(
				boost::asio::null_buffers(),
				boost::bind(&HalfDuplexTcpSocketConnection::completeSending, shared_from_this(), boost::asio::placeholders::error)
			);
		}
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
			doStartOperation();
		// otherwise, an error occurred. Closing the socket cancels any outstanding asynchronous read or write operations.
		// the FullDuplexTcpSocketConnection object will be destroyed automatically once those outstanding operations complete.
		else
			socket_.close();
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
			doStartOperation();
		// otherwise, an error occurred. Closing the socket cancels any outstanding asynchronous read or write operations.
		// the HalfDuplexTcpSocketConnection object will be destroyed automatically once those outstanding operations complete.
		else
			socket_.close();
	}

private:
	boost::asio::io_service &ioService_;
	boost::asio::ip::tcp::socket socket_;

	session_type session_;

	bool isReceiving_;
	bool isSending_;
};

}  // namespace swl


#endif  // __SWL_UTIL__TCP_SOCKET_CONNECTION__H_
