#if !defined(__SWL_UTIL__HALF_DUPLEX_TCP_SOCKET_SESSION__H_)
#define __SWL_UTIL__HALF_DUPLEX_TCP_SOCKET_SESSION__H_ 1


#include "swl/util/ExportUtil.h"
#include "swl/util/GuardedBuffer.h"
#include <boost/asio.hpp>


namespace swl {

//-----------------------------------------------------------------------------------
//

/**
 *	@brief  half duplex mode를 이용하여 TCP socket 통신을 수행하는 server session class.
 *
 *	내부적으로 asynchronous I/O를 사용하고 있으므로 이 class를 사용하는 S/W에 부담을 적게 주는 장점이 있다.
 *
 *	half duplex mode로 통신을 수행하므로 송수신이 반복적으로 수행되어야 한다.
 *	따라서 아래와 같이 send() & receive() 함수가 번갈아 호출되어야 한다.
 *		- case 1
 *			-# send()
 *			-# receive()
 *			-# send()
 *			-# ...
 *		- case 2
 *			-# receive()
 *			-# send()
 *			-# receive()
 *			-# ...
 */
class SWL_UTIL_API HalfDuplexTcpSocketSession
{
public:
	//typedef HalfDuplexTcpSocketSession base_type;

public:
	/**
	 *	@brief  [ctor] contructor.
	 *	@param[in]  ioService  TCP socket 통신을 위한 Boost.ASIO의 I/O service 객체.
	 *
	 *	TCP socket 통신 session을 위해 필요한 설정들을 초기화한다.
	 */
	HalfDuplexTcpSocketSession(boost::asio::ip::tcp::socket &socket);
	/**
	 *	@brief  [dtor] virtual default destructor.
	 *
	 *	TCP socket 통신 session을 종료하기 위해 필요한 절차를 수행한다.
	 */
	virtual ~HalfDuplexTcpSocketSession();

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
	 *	@brief  TCP socket 통신 중에 발생한 송신 오류의 error code를 반환.
	 *	@param[out]  ec  발생한 오류의 error code 객체.
	 *
	 *	TCP socket 통신을 통해 message를 전송하는 동안 발생한 오류의 error code를 반환한다.
	 */
	virtual void send(boost::system::error_code &ec);

	/**
	 *	@brief  TCP socket session의 message 수신 준비 상태를 확인.
	 *	@return  TCP socket session이 수신 가능 상태라면 true 반환.
	 *
	 *	TCP socket session이 수신 가능 상태에 있다면 true를, 그렇지 않다면 false를 반환한다.
	 */
	bool isReadyToReceive() const
	{  return state_ == RECEIVING;  }

	/**
	 *	@brief  TCP socket 통신 중에 발생한 수신 오류의 error code를 반환.
	 *	@param[out]  ec  발생한 오류의 error code 객체.
	 *
	 *	TCP socket 통신을 통해 message를 수신하는 동안 발생한 오류의 error code를 반환한다.
	 */
	virtual void receive(boost::system::error_code &ec);

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
	size_t getSendBufferSize() const;
	/**
	 *	@brief  TCP socket 통신을 통해 수신된 message의 길이를 반환.
	 *	@return  수신된 message의 길이를 반환.
	 *
	 *	TCP socket 통신을 통해 수신된 message를 저장하고 있는 수신 buffer의 길이를 반환한다.
	 */
	size_t getReceiveBufferSize() const;

private:
	static const std::size_t MAX_SEND_LENGTH_ = 512;
	static const std::size_t MAX_RECEIVE_LENGTH_ = 512;

	boost::asio::ip::tcp::socket &socket_;
	enum { SENDING, RECEIVING } state_;

	GuardedByteBuffer sendBuffer_;
	GuardedByteBuffer receiveBuffer_;
	GuardedByteBuffer::value_type sendMsg_[MAX_SEND_LENGTH_];
	GuardedByteBuffer::value_type receiveMsg_[MAX_RECEIVE_LENGTH_];
	std::size_t sentMsgLength_;
};

}  // namespace swl


#endif  // __SWL_UTIL__HALF_DUPLEX_TCP_SOCKET_SESSION__H_
