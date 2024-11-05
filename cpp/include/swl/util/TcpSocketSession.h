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
 *	@brief  TCP socket ����� �����ϴ� server session class.
 *
 *	TCP socket server�� connection ��ü (����: TcpSocketConnectionUsingSession) ���� ����ϱ� ���� ����� session class�̴�.
 */
class SWL_UTIL_API TcpSocketSession
{
public:
	//typedef TcpSocketSession base_type;

protected:
	/**
	 *	@brief  [ctor] constructor.
	 *	@param[in]  socket  TCP socket ����� ���� Boost.ASIO�� socket ��ü.
	 *
	 *	TCP socket ��� session�� ���� �ʿ��� �������� �ʱ�ȭ�Ѵ�.
	 */
	TcpSocketSession(boost::asio::ip::tcp::socket &socket);
public:
	/**
	 *	@brief  [dtor] virtual default destructor.
	 *
	 *	TCP socket ��� session�� �����ϱ� ���� �ʿ��� ������ �����Ѵ�.
	 */
	virtual ~TcpSocketSession();

public:
	/**
	 *	@brief  TCP socket session�� message ���� �غ� ���¸� Ȯ��.
	 *	@return  TCP socket session�� ���� ���� ���¶�� true ��ȯ.
	 *
	 *	TCP socket session�� ���� ���� ���¿� �ִٸ� true��, �׷��� �ʴٸ� false�� ��ȯ�Ѵ�.
	 */
	bool isReadyToSend() const
	{  return state_ == SENDING;  }

	/**
	 *	@brief  send buffer�� �ִ� message�� ����� TCP socket ��� channel�� ���� ����.
	 *	@param[out]  ec  message ���� �������� �߻��� ������ error code ��ü.
	 *
	 *	��û�� message�� TCP socket ����� ���� �����Ѵ�.
	 *
	 *	TCP socket ����� ���� message�� �����ϴ� ���� �߻��� ������ error code�� ���ڷ� �Ѿ�´�.
	 */
	virtual void send(boost::system::error_code &ec) = 0;

	/**
	 *	@brief  TCP socket session�� message ���� �غ� ���¸� Ȯ��.
	 *	@return  TCP socket session�� ���� ���� ���¶�� true ��ȯ.
	 *
	 *	TCP socket session�� ���� ���� ���¿� �ִٸ� true��, �׷��� �ʴٸ� false�� ��ȯ�Ѵ�.
	 */
	bool isReadyToReceive() const
	{  return state_ == RECEIVING;  }

	/**
	 *	@brief  TCP socket ��� channel�� ���� ���ŵ� message�� receive buffer�� ����.
	 *	@param[out]  ec  message ���� �������� �߻��� ������ error code ��ü.
	 *
	 *	TCP socket ����� ���� ���ŵ� message�� receive buffer�� �����Ѵ�.
	 *
	 *	TCP socket ����� ���� message�� �����ϴ� ���� �߻��� ������ error code�� ���ڷ� �Ѿ�´�.
	 */
	virtual void receive(boost::system::error_code &ec) = 0;

	/**
	 *	@brief  TCP socket ����� �۽� buffer�� ���.
	 *
	 *	���۵��� ���� �۽� buffer�� ��� message�� �����Ѵ�.
	 *	������ �۽� message�� ������ �������� �� �� �����Ƿ� ����ġ ���� error�� �߻���ų �� �ִ�.
	 */
	void clearSendBuffer();
	/**
	 *	@brief  TCP socket ����� ���� buffer�� ���.
	 *
	 *	TCP socket ��� channel�� ���ŵ� ���� buffer�� ��� message�� �����Ѵ�.
	 *	������ ���� message�� ������ �������� �� �� �����Ƿ� ����ġ ���� error�� �߻���ų �� �ִ�.
	 */
	void clearReceiveBuffer();

	/**
	 *	@brief  TCP socket ��� channel�� �۽� buffer�� ��� �ִ����� Ȯ��.
	 *	@return  �۽� buffer�� ��� �ִٸ� true�� ��ȯ.
	 *
	 *	TCP socket ����� ���� ������ message�� �۽� buffer�� ��� �ִ��� ���θ� ��ȯ�Ѵ�.
	 */
	bool isSendBufferEmpty() const;
	/**
	 *	@brief  TCP socket ��� channel�� ���� buffer�� ��� �ִ����� Ȯ��.
	 *	@return  ���� buffer�� ��� �ִٸ� true�� ��ȯ.
	 *
	 *	TCP socket ����� ���� ���ŵ� message�� ���� buffer�� ��� �ִ��� ���θ� ��ȯ�Ѵ�.
	 */
	bool isReceiveBufferEmpty() const;

	/**
	 *	@brief  TCP socket ����� ���� �۽��� message�� ���̸� ��ȯ.
	 *	@return  �۽� message�� ���̸� ��ȯ.
	 *
	 *	TCP socket ����� ���� ������ message�� �����ϰ� �ִ� �۽� buffer�� ���̸� ��ȯ�Ѵ�.
	 */
	std::size_t getSendBufferSize() const;
	/**
	 *	@brief  TCP socket ����� ���� ���ŵ� message�� ���̸� ��ȯ.
	 *	@return  ���ŵ� message�� ���̸� ��ȯ.
	 *
	 *	TCP socket ����� ���� ���ŵ� message�� �����ϰ� �ִ� ���� buffer�� ���̸� ��ȯ�Ѵ�.
	 */
	std::size_t getReceiveBufferSize() const;

protected:
	/**
	 *	@brief  �� ���� �۽� �������� ���� �� �ִ� message�� �ִ� ����.
	 */
	static const std::size_t MAX_SEND_LENGTH_ = 512;
	/**
	 *	@brief  �� ���� ���� �������� ���� �� �ִ� message�� �ִ� ����.
	 */
	static const std::size_t MAX_RECEIVE_LENGTH_ = 512;

	/**
	 *	@brief  TCP socket ����� ���������� �����ϴ� Boost.ASIO�� socket ��ü.
	 */
	boost::asio::ip::tcp::socket &socket_;
	/**
	 *	@brief  TCP socket�� �۽� ������ �ִ��� ���� ������ �ִ����� �����ϴ� state ����.
	 */
	enum { SENDING, RECEIVING } state_;

	/**
	 *	@brief  TCP socket ����� ���� send buffer.
	 *
	 *	GuardedByteBuffer�� ��ü�� multi-thread ȯ�濡���� �����ϰ� ����� �� �ִ�.
	 */
	GuardedByteBuffer sendBuffer_;
	/**
	 *	@brief  TCP socket ����� ���� send buffer.
	 *
	 *	GuardedByteBuffer�� ��ü�� multi-thread ȯ�濡���� �����ϰ� ����� �� �ִ�.
	 */
	GuardedByteBuffer receiveBuffer_;
	/**
	 *	@brief  �� ���� �۽� �������� �����ϰ� �� message�� �����ϴ� buffer.
	 *
	 *	buffer�� ���̴� MAX_SEND_LENGTH_�̴�.
	 */
	boost::array<GuardedByteBuffer::value_type, MAX_SEND_LENGTH_> sendMsg_;
	/**
	 *	@brief  �� ���� ���� �������� �����ϰ� �� message�� �����ϴ� buffer.
	 *
	 *	buffer�� ���̴� MAX_RECEIVE_LENGTH_�̴�.
	 */
	boost::array<GuardedByteBuffer::value_type, MAX_RECEIVE_LENGTH_> receiveMsg_;
};

}  // namespace swl


#endif  // __SWL_UTIL__TCP_SOCKET_SESSION__H_
