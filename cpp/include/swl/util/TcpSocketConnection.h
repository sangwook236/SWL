#if !defined(__SWL_UTIL__TCP_SOCKET_CONNECTION__H_)
#define __SWL_UTIL__TCP_SOCKET_CONNECTION__H_ 1


#include "swl/util/ExportUtil.h"
#include "swl/util/GuardedBuffer.h"
#include <boost/asio.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/bind.hpp>
#include <boost/array.hpp>


namespace swl {

//-----------------------------------------------------------------------------------
//

/**
 *	@brief  asynchronous I/O mode로 동작하는 TCP socket server의 channel 연결을 처리하는 connection utility class.
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
class SWL_UTIL_API TcpSocketConnection: public boost::enable_shared_from_this<TcpSocketConnection>
{
public:
	typedef boost::enable_shared_from_this<TcpSocketConnection> base_type;
	typedef boost::shared_ptr<TcpSocketConnection> pointer;

protected:
	/**
	 *	@brief  [ctor] protected constructor.
	 *	@param[in]  ioService  TCP socket 통신을 위한 Boost.ASIO의 I/O service 객체.
	 *
	 *	TCP socket connection 객체의 초기화를 수행한다.
	 */
	TcpSocketConnection(boost::asio::io_service &ioService);
public:
	/**
	 *	@brief  [dtor] virtual default destructor.
	 *
	 *	TCP socket 통신 connection을 종료하기 위한 작업을 수행한다.
	 */
	virtual ~TcpSocketConnection();

private:
	/**
	 *	@brief  [ctor] TCP socket connection 객체의 생성을 위한 factory 함수.
	 *	@param[in]  ioService  TCP socket 통신을 위한 Boost.ASIO의 I/O service 객체.
	 *
	 *	TCP socket connection 객체의 instance를 생성한다.
	 */
	static pointer create(boost::asio::io_service &ioService);

public:
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
	void start();

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
	 *	@brief  TCP socket 통신 channel의 송신 buffer가 비어 있는지를 확인.
	 *	@return  송신 buffer가 비어 있다면 true를 반환.
	 *
	 *	TCP socket 통신을 통해 전송할 message의 송신 buffer가 비어 있는지 여부를 반환한다.
	 */
	bool isSendBufferEmpty() const;
	/**
	 *	@brief  TCP socket 통신 channel의 수신 buffer가 비어 있는지를 확인.
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
	 *	@brief  송신 buffer에 저장된 message의 전송을 시작.
	 *
	 *	송신 buffer에 저장되어 있는 message를 asynchronous I/O를 통해 송신한다.
	 */
	virtual void doStartOperation() = 0;
	/**
	 *	@brief  송신 요청된 message의 전송이 완료된 경우 호출되는 completion routine.
	 *	@param[in]  ec  message를 전송하는 과정에서 발생한 오류의 error code.
	 *	@throw  LogException  TCP socket의 close 과정에서 error가 발생.
	 *
	 *	asynchronous I/O를 이용하여 송신 요청된 message의 전송이 완료되었을 때 system에 의해 호출되는 completion routine이다.
	 *	doStartOperation() 함수 내에서 asynchronous 송신 요청을 하면서 해당 함수를 completion routine으로 지정해 주어야 한다.
	 */
	virtual void doCompleteSending(boost::system::error_code ec) = 0;
	/**
	 *	@brief  TCP socket 통신 channel을 통해 수신된 message가 있는 경우 호출되는 completion routine.
	 *	@param[in]  ec  message를 수신하는 과정에서 발생한 오류의 error code.
	 *	@param[in]  bytesTransferred  수신된 message의 길이.
	 *	@throw  LogException  TCP socket의 close 과정에서 error가 발생.
	 *
	 *	asynchronous I/O를 통해 message의 수신되는 경우 system에 의해 호출되는 completion routine이다.
	 *	doStartOperation() 함수 내에서 asynchronous 수신 요청을 하면서 해당 함수를 completion routine으로 지정해 주어야 한다.
	 */
	virtual void doCompleteReceiving(boost::system::error_code ec, std::size_t bytesTransferred) = 0;

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

	/**
	 *	@brief  TCP socket 통신을 통해 message 송신 중인지를 확인.
	 */
	bool isSending_;
	/**
	 *	@brief  TCP socket 통신을 통해 message 수신 중인지를 확인.
	 */
	bool isReceiving_;
};

}  // namespace swl


#endif  // __SWL_UTIL__TCP_SOCKET_CONNECTION__H_
