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
 *	@brief  asynchronous I/O mode�� �����ϴ� TCP socket server�� channel ������ ó���ϴ� connection utility class.
 *
 *	server���� client�� ���� ��û�� accept�� ��, TCP socket connection ��ü�� ���ӵ� client�� ���� ���� �� ó���� �����ϰ� �Ѵ�.
 *	�������� ������ �Ʒ��� ����.
 *		-# [client] ���� ��û
 *		-# [server] client�κ��� ���� ��û ����
 *		-# [server] connection ��ü�� client�� ����
 *		-# [connection] socket ����� �̿��� client�� message �ۼ��� ����
 *		-# [server] �ٸ� client�� ���� ��û ���
 *
 *	TCP socket ����� asynchronous I/O�� �̿��Ͽ� �����Ѵ�.
 */
class SWL_UTIL_API TcpSocketConnection: public boost::enable_shared_from_this<TcpSocketConnection>
{
public:
	typedef boost::enable_shared_from_this<TcpSocketConnection> base_type;
	typedef boost::shared_ptr<TcpSocketConnection> pointer;

protected:
	/**
	 *	@brief  [ctor] protected constructor.
	 *	@param[in]  ioService  TCP socket ����� ���� Boost.ASIO�� I/O service ��ü.
	 *
	 *	TCP socket connection ��ü�� �ʱ�ȭ�� �����Ѵ�.
	 */
	TcpSocketConnection(boost::asio::io_service &ioService);
public:
	/**
	 *	@brief  [dtor] virtual default destructor.
	 *
	 *	TCP socket ��� connection�� �����ϱ� ���� �۾��� �����Ѵ�.
	 */
	virtual ~TcpSocketConnection();

private:
	/**
	 *	@brief  [ctor] TCP socket connection ��ü�� ������ ���� factory �Լ�.
	 *	@param[in]  ioService  TCP socket ����� ���� Boost.ASIO�� I/O service ��ü.
	 *
	 *	TCP socket connection ��ü�� instance�� �����Ѵ�.
	 */
	static pointer create(boost::asio::io_service &ioService);

public:
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
	void start();

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
	 *	@brief  �۽� buffer�� ����� message�� ������ ����.
	 *
	 *	�۽� buffer�� ����Ǿ� �ִ� message�� asynchronous I/O�� ���� �۽��Ѵ�.
	 */
	virtual void doStartOperation() = 0;
	/**
	 *	@brief  �۽� ��û�� message�� ������ �Ϸ�� ��� ȣ��Ǵ� completion routine.
	 *	@param[in]  ec  message�� �����ϴ� �������� �߻��� ������ error code.
	 *	@throw  LogException  TCP socket�� close �������� error�� �߻�.
	 *
	 *	asynchronous I/O�� �̿��Ͽ� �۽� ��û�� message�� ������ �Ϸ�Ǿ��� �� system�� ���� ȣ��Ǵ� completion routine�̴�.
	 *	doStartOperation() �Լ� ������ asynchronous �۽� ��û�� �ϸ鼭 �ش� �Լ��� completion routine���� ������ �־�� �Ѵ�.
	 */
	virtual void doCompleteSending(boost::system::error_code ec) = 0;
	/**
	 *	@brief  TCP socket ��� channel�� ���� ���ŵ� message�� �ִ� ��� ȣ��Ǵ� completion routine.
	 *	@param[in]  ec  message�� �����ϴ� �������� �߻��� ������ error code.
	 *	@param[in]  bytesTransferred  ���ŵ� message�� ����.
	 *	@throw  LogException  TCP socket�� close �������� error�� �߻�.
	 *
	 *	asynchronous I/O�� ���� message�� ���ŵǴ� ��� system�� ���� ȣ��Ǵ� completion routine�̴�.
	 *	doStartOperation() �Լ� ������ asynchronous ���� ��û�� �ϸ鼭 �ش� �Լ��� completion routine���� ������ �־�� �Ѵ�.
	 */
	virtual void doCompleteReceiving(boost::system::error_code ec, std::size_t bytesTransferred) = 0;

protected:
	/**
	 *	@brief  �� ���� �۽� �������� ���� �� �ִ� message�� �ִ� ����.
	 */
#if defined(__GNUC__)
	static const unsigned long MAX_SEND_LENGTH_ = 512;
#else
	static const std::size_t MAX_SEND_LENGTH_ = 512;
#endif
	/**
	 *	@brief  �� ���� ���� �������� ���� �� �ִ� message�� �ִ� ����.
	 */
#if defined(__GNUC__)
	static const unsigned long MAX_RECEIVE_LENGTH_ = 512;
#else
	static const std::size_t MAX_RECEIVE_LENGTH_ = 512;
#endif

	/**
	 *	@brief  TCP socket ����� ���������� �����ϴ� Boost.ASIO�� socket ��ü.
	 */
	boost::asio::ip::tcp::socket socket_;

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
	/**
	 *	@brief  ���� �ֱ� �۽� �������� ������ message�� ����.
	 */
	std::size_t sentMsgLength_;

	/**
	 *	@brief  TCP socket ����� ���� message �۽� �������� Ȯ��.
	 */
	bool isSending_;
	/**
	 *	@brief  TCP socket ����� ���� message ���� �������� Ȯ��.
	 */
	bool isReceiving_;
};

}  // namespace swl


#endif  // __SWL_UTIL__TCP_SOCKET_CONNECTION__H_
