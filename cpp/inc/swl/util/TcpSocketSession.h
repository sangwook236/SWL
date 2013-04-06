#if !defined(__SWL_UTIL__TCP_SOCKET_SESSION__H_)
#define __SWL_UTIL__TCP_SOCKET_SESSION__H_ 1


#include "swl/util/ExportUtil.h"
#include "swl/util/GuardedBuffer.h"
#include <boost/asio.hpp>
#include <boost/array.hpp>


namespace swl {

//-----------------------------------------------------------------------------------
//

/**
 *	@brief  TCP socket 통신을 수행하는 server session class.
 *
 *	TCP socket server의 connection 객체 (참고: TcpSocketConnectionUsingSession) 에서 사용하기 위해 설계된 session class이다.
 */
class SWL_UTIL_API TcpSocketSession
{
public:
	//typedef TcpSocketSession base_type;

protected:
	/**
	 *	@brief  [ctor] constructor.
	 *	@param[in]  socket  TCP socket 통신을 위한 Boost.ASIO의 socket 객체.
	 *
	 *	TCP socket 통신 session을 위해 필요한 설정들을 초기화한다.
	 */
	TcpSocketSession(boost::asio::ip::tcp::socket &socket);
public:
	/**
	 *	@brief  [dtor] virtual default destructor.
	 *
	 *	TCP socket 통신 session을 종료하기 위해 필요한 절차를 수행한다.
	 */
	virtual ~TcpSocketSession();

public:
	/**
	 *	@brief  TCP socket session의 message 전송 준비 상태를 확인.
	 *	@return  TCP socket session이 전송 가능 상태라면 true 반환.
	 *
	 *	TCP socket session이 전송 가능 상태에 있다면 true를, 그렇지 않다면 false를 반환한다.
	 */
	bool isReadyToSend() const
	{  return state_ == SENDING;  }

	/**
	 *	@brief  send buffer에 있는 message를 연결된 TCP socket 통신 channel을 통해 전송.
	 *	@param[out]  ec  message 전송 과정에서 발생한 오류의 error code 객체.
	 *
	 *	요청된 message를 TCP socket 통신을 통해 전송한다.
	 *
	 *	TCP socket 통신을 통해 message를 전송하는 동안 발생한 오류의 error code를 인자로 넘어온다.
	 */
	virtual void send(boost::system::error_code &ec) = 0;

	/**
	 *	@brief  TCP socket session의 message 수신 준비 상태를 확인.
	 *	@return  TCP socket session이 수신 가능 상태라면 true 반환.
	 *
	 *	TCP socket session이 수신 가능 상태에 있다면 true를, 그렇지 않다면 false를 반환한다.
	 */
	bool isReadyToReceive() const
	{  return state_ == RECEIVING;  }

	/**
	 *	@brief  TCP socket 통신 channel을 통해 수신된 message를 receive buffer에 저장.
	 *	@param[out]  ec  message 수신 과정에서 발생한 오류의 error code 객체.
	 *
	 *	TCP socket 통신을 통해 수신된 message를 receive buffer에 저장한다.
	 *
	 *	TCP socket 통신을 통해 message를 수신하는 동안 발생한 오류의 error code를 인자로 넘어온다.
	 */
	virtual void receive(boost::system::error_code &ec) = 0;

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
	boost::asio::ip::tcp::socket &socket_;
	/**
	 *	@brief  TCP socket이 송신 과저에 있는지 수신 과정에 있는지를 저장하는 state 변수.
	 */
	enum { SENDING, RECEIVING } state_;

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
};

}  // namespace swl


#endif  // __SWL_UTIL__TCP_SOCKET_SESSION__H_
