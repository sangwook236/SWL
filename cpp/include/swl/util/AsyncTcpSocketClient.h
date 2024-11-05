#if !defined(__SWL_UTIL__ASYNC_TCP_SOCKET_CLIENT__H_)
#define __SWL_UTIL__ASYNC_TCP_SOCKET_CLIENT__H_ 1


#include "swl/util/ExportUtil.h"
#include "swl/util/GuardedBuffer.h"
#include <boost/asio.hpp>
#include <boost/array.hpp>
#include <string>


namespace swl {

//-----------------------------------------------------------------------------------
//

/**
 *	@brief  �񵿱������� TCP socket ����� �����ϴ� client class.
 *
 *	TCP socket ����� ���� message�� �ۼ����ϱ� ���� send() �Լ��� receive() �Լ��� ȣ���ϸ� �ȴ�.
 *	TCP socket ����� �����ϴ� �������� ������ �Ʒ��� ����.
 *		- AsyncTcpSocketClient ��ü ����
 *		- doStartConnecting() �Լ��� �̿��Ͽ� TCP server�� ����
 *		- send() and/or receive() �Լ��� �̿��� message �ۼ���
 *		- �۾��� �����ٸ�, disconnect() �Լ��� ȣ���Ͽ� ���� ����
 *		- AsyncTcpSocketClient ��ü �Ҹ�
 *
 *	asynchronous I/O�� ����Ͽ� �ۼ����� �����Ѵ�.
 */
class SWL_UTIL_API AsyncTcpSocketClient
{
public:
	//typedef AsyncTcpSocketClient base_type;

public:
	/**
	 *	@brief  [ctor] constructor.
	 *	@param[in]  ioService  TCP socket ����� ���� Boost.ASIO�� I/O service ��ü.
	 *	@param[in]  hostName  TCP socket server�� host �̸�.
	 *	@param[in]  serviceName  TCP socket server�� service �̸�.
	 *
	 *	TCP socket ����� ���� �ʿ��� �������� �ʱ�ȭ�ϰ�,
	 *	���ڷ� �Ѱ��� host �̸��� service �̸��� �̿��Ͽ� TCP socket channel�� �����Ѵ�.
	 *
	 *	host �̸��� IP address�̳� domain �̸����� ������ �� �ִ�.
	 *		- "abc.com"
	 *		- "100.110.120.130"
	 *	service �̸��� �̳� port ��ȣ�� ������ �� �ִ�.
	 *		- "http" or "daytime"
	 *		- "80"
	 */
#if defined(_UNICODE) || defined(UNICODE)
	AsyncTcpSocketClient(boost::asio::io_service &ioService, const std::wstring &hostName, const std::wstring &serviceName);
#else
	AsyncTcpSocketClient(boost::asio::io_service &ioService, const std::string &hostName, const std::string &serviceName);
#endif
	/**
	 *	@brief  [dtor] virtual default destructor.
	 *
	 *	TCP socket ����� �����ϱ� ���� ������ �����Ѵ�.
	 *	��� channel�� ���� �ִ� ��� disconnect() �Լ��� ȣ���Ͽ� �̸� �ݴ´�.
	 */
	virtual ~AsyncTcpSocketClient();

public:
	/**
	 *	@brief  TCP socket ��� channel�� ������ ����.
	 *	@throw  LogException  TCP socket�� close �������� error�� �߻�.
	 *
	 *	TCP socket ��� channel�� ������ ����, ��� ���� resource�� ��ȯ�Ѵ�.
	 */
	void disconnect();

	/**
	 *	@brief  TCP socket ����� ���� ���¿� �ִ��� Ȯ��.
	 *	@return  TCP socket ��� channel�� ���� �����̸� true ��ȯ.
	 *
	 *	TCP socket ��� channel�� ���� ���¸� ��ȯ�Ѵ�.
	 */
	bool isConnected() const  {  return isActive_;  }

	/**
	 *	@brief  ������ message�� ����� TCP socket ��� channel�� ���� ����.
	 *	@param[in]  msg  ������ message�� �����ϴ� pointer.
	 *	@param[in]  len  ������ message ����.
	 *
	 *	��û�� message�� TCP socket ����� ���� �����Ѵ�.
	 *	asynchronous I/O�� ���� message�� �����Ѵ�.
	 */
	void send(const unsigned char *msg, const std::size_t len);
	/**
	 *	@brief  TCP socket ��� channel�� ���� message�� ����.
	 *	@param[out]  msg  ���ŵ� message�� ������ pointer.
	 *	@param[in]  len  asynchronous I/O�� ���� ������ message�� ������ buffer�� ũ�⸦ ����.
	 *	@return  ������ ���ŵ� message�� ���̸� ��ȯ. ���ڷ� ������ len���� �۰ų� ����.
	 *
	 *	TCP socket ����� ���� ���ŵǴ� message�� ���ڷ� ������ pointer�� ��ü�� �����Ѵ�.
	 *	asynchronous I/O�� ���� message�� �����Ѵ�.
	 */
	std::size_t receive(unsigned char *msg, const std::size_t len);

	/**
	 *	@brief  ���� ���� I/O �۾��� ���.
	 *	@throw  LogException  �ۼ��� operation�� ����ϴ� �������� error�� �߻�.
	 *
	 *	asynchronous I/O�� ���� ���� ���� �ۼ��� operation�� ����Ѵ�.
	 */
	void cancelIo();

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
	 *	@brief  TCP socket ����� �۽� buffer�� ��� �ִ����� Ȯ��.
	 *	@return  �۽� buffer�� ��� �ִٸ� true�� ��ȯ.
	 *
	 *	TCP socket ����� ���� ������ message�� �۽� buffer�� ��� �ִ��� ���θ� ��ȯ�Ѵ�.
	 */
	bool isSendBufferEmpty() const;
	/**
	 *	@brief  TCP socket ����� ���� buffer�� ��� �ִ����� Ȯ��.
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
	 *	@brief  asynchronous mode�� ������ host �̸��� service �̸��� �̿��� TCP socket server�� channel�� ����.
	 *	@param[in]  endpoint_iterator  TCP socket ����� ���� ������ server�� end point.
	 *
	 *	���ڷ� �Ѱ��� host �̸��� service �̸��� �̿��Ͽ� TCP socket channel�� �����Ѵ�.
	 */
	virtual void doStartConnecting(boost::asio::ip::tcp::resolver::iterator endpoint_iterator);
	/**
	 *	@brief  asynchronous I/O�� ���� ��û�� server ���� ��û�� �Ϸ�� ��� ȣ��Ǵ� completion routine.
	 *	@param[in]  ec  server ���� �������� �߻��� ������ error code.
	 *	@param[in]  endpoint_iterator  TCP socket ����� ���� ������ server�� end point.
	 *
	 *	asynchronous I/O�� �̿��Ͽ� ��û�� ���� �õ��� �Ϸ�Ǿ��� �� system�� ���� ȣ��Ǵ� completion routine�̴�.
	 *	������ ���������� �Ϸ�Ǿ��ٸ� isConnected()�� true�� ��ȯ�ϰ� �ȴ�.
	 */
	virtual void doCompleteConnecting(const boost::system::error_code &ec, boost::asio::ip::tcp::resolver::iterator endpoint_iterator);
	/**
	 *	@brief  �۽� buffer�� ����� message�� ������ ����.
	 *
	 *	�۽� buffer�� ����Ǿ� �ִ� message�� asynchronous I/O�� ���� �۽��Ѵ�.
	 */
	virtual void doStartSending();
	/**
	 *	@brief  �۽� ��û�� message�� ������ �Ϸ�� ��� ȣ��Ǵ� completion routine.
	 *	@param[in]  ec  message�� �����ϴ� �������� �߻��� ������ error code.
	 *	@throw  LogException  TCP socket channel�� close �������� error�� �߻�.
	 *
	 *	asynchronous I/O�� �̿��Ͽ� �۽� ��û�� message�� ������ �Ϸ�Ǿ��� �� system�� ���� ȣ��Ǵ� completion routine�̴�.
	 *	doStartSending() �Լ� ������ asynchronous �۽� ��û�� �ϸ鼭 �ش� �Լ��� completion routine���� ������ �־�� �Ѵ�.
	 */
	virtual void doCompleteSending(const boost::system::error_code &ec);
	/**
	 *	@brief  TCP socket ��� channel�� ���� ������ message�� receive buffer�� ���� ����.
	 *
	 *	TCP socket ����� ���ŵǴ� message�� asynchronous I/O�� �̿��Ͽ� �����ϱ� �����Ѵ�.
	 */
	virtual void doStartReceiving();
	/**
	 *	@brief  TCP socket ��� channel�� ���� ���ŵ� message�� �ִ� ��� ȣ��Ǵ� completion routine.
	 *	@param[in]  ec  message�� �����ϴ� �������� �߻��� ������ error code.
	 *	@param[in]  bytesTransferred  ���ŵ� message�� ����.
	 *	@throw  LogException  TCP socket channel�� close �������� error�� �߻�.
	 *
	 *	asynchronous I/O�� ���� message�� ���ŵǴ� ��� system�� ���� ȣ��Ǵ� completion routine�̴�.
	 *	doStartReceiving() �Լ� ������ asynchronous ���� ��û�� �ϸ鼭 �ش� �Լ��� completion routine���� ������ �־�� �Ѵ�.
	 */
	virtual void doCompleteReceiving(const boost::system::error_code &ec, std::size_t bytesTransferred);

private:
	void doSendOperation(const unsigned char *msg, const std::size_t len);
	void doCloseOperation(const boost::system::error_code &ec);
	void doCancelOperation(const boost::system::error_code &ec);

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
	 *	@brief  TCP socket ��� channel�� ����Ǿ� �ְ� ���� ���������� Ȯ���ϴ� flag ����.
	 *
	 *	TCP socket ��� channel�� ����Ǿ� �ְ� ���� ���¶�� true��, �׷��� �ʴٸ� false�� ǥ���Ѵ�.
	 */
	bool isActive_;

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
};

}  // namespace swl


#endif  // __SWL_UTIL__ASYNC_TCP_SOCKET_CLIENT__H_
