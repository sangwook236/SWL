#if !defined(__SWL_UTIL__SERIAL_PORT__H_)
#define __SWL_UTIL__SERIAL_PORT__H_ 1


#include "swl/util/ExportUtil.h"
#include "swl/util/GuardedBuffer.h"
#include <boost/asio.hpp>
#include <boost/array.hpp>
#include <string>


namespace swl {

//-----------------------------------------------------------------------------------
//

/**
 *	@brief  serial ����� �����ϱ� ���� class.
 *
 *	serial ����� ���� �ʿ��� ������ �Ʒ��� ����.
 *		- ��� port: connect() �Լ����� ����
 *		- Baud rate: connect() �Լ����� ����
 *		- data bit: 8 bits
 *		- stop bit: 1 bit
 *		- parity: none
 *		- H/W handshaking: ��� ����
 *
 *	���������� asynchronous I/O�� ����ϰ� �����Ƿ� �� class�� ����ϴ� S/W�� �δ��� ���� �ִ� ������ �ִ�.
 *
 *	serial ����� ���� message�� ������ ��쿡�� send() �Լ��� �ش� ������ ���� ȣ���ϸ� �ǰ�,
 *	������ ��쿡�� receive() �Լ��� ȣ���ϸ� �ȴ�.
 *
 *	serial ����� ������ �����ϱ� ���ؼ��� connect()�� �Լ��� ȣ���Ͽ� port�� ������ ��
 *	constructor�� ���ڷ� �Ѱ��� I/O�� run���Ѿ� �Ѵ�.
 */
class SWL_UTIL_API SerialPort
{
public:
	//typedef SerialPort base_type;

public:
	/**
	 *	@brief  [ctor] constructor.
	 *	@param[in]  ioService  serial ����� ���� Boost.ASIO�� I/O service ��ü.
	 *
	 *	serial ����� ���� �ʿ��� �������� �ʱ�ȭ�Ѵ�.
	 */
	SerialPort(boost::asio::io_service &ioService);
	/**
	 *	@brief  [dtor] virtual default destructor.
	 *
	 *	serial ����� �����ϱ� ���� �ʿ��� ������ �����Ѵ�.
	 *	��� port�� ���� �ִ� ��� disconnect() �Լ��� ȣ���Ͽ� �̸� �ݴ´�.
	 */
	virtual ~SerialPort();

public:
	/**
	 *	@brief  ������ COM port�� Baud rate�� serial ��� channel�� ����.
	 *	@param[in]  portName  serial ����� ���� port �̸�.
	 *	@param[in]  baudRate  ����� ���� ����ϰ��� �ϴ� �ӵ�.
	 *	@return  serial ��� channel�� ���������� ����Ǿ��ٸ� true ��ȯ.
	 *
	 *	���ڷ� �Ѱ��� port �̸��� Baud rate�� �̿��Ͽ� ��� ä���� �����ϰ�
	 *	asynchronous I/O �۾��� �����ϱ� ���� �۾��� �����Ѵ�.
	 */
#if defined(_UNICODE) || defined(UNICODE)
	bool connect(const std::wstring &portName, const unsigned int baudRate);
#else
	bool connect(const std::string &portName, const unsigned int baudRate);
#endif
	/**
	 *	@brief  serial ����� ���� port�� ����.
	 *	@throw  LogException  serial port�� close �������� error�� �߻�.
	 *
	 *	serial ����� ���� �����Ͽ��� channel�� ����, Ȱ���� resource�� ��ȯ�Ѵ�.
	 */
	void disconnect();

	/**
	 *	@brief  serial ����� ���� port�� ����Ǿ� �ִ��� Ȯ��.
	 *	@return  serial ��� channel�� ����Ǿ� �ִٸ� true ��ȯ.
	 *
	 *	serial ��� channel�� ���� ���¸� ��ȯ�Ѵ�.
	 */
	bool isConnected() const  {  return isActive_;  }

	/**
	 *	@brief  ������ message�� ����� serial ��� channel�� ���� ����.
	 *	@param[in]  msg  ������ message�� �����ϴ� pointer.
	 *	@param[in]  len  ������ message ����.
	 *
	 *	��û�� message�� serial ����� ���� �����Ѵ�.
	 *	asynchronous I/O�� ���� message�� �����Ѵ�.
	 */
	void send(const unsigned char *msg, const std::size_t len);
	/**
	 *	@brief  ����� serial ��� channel�� ���� message�� ����.
	 *	@param[out]  msg  ���ŵ� message�� ������ pointer.
	 *	@param[in]  len  asynchronous I/O�� ���� ������ message�� ������ buffer�� ũ�⸦ ����.
	 *	@return  ������ ���ŵ� message�� ���̸� ��ȯ. ���ڷ� ������ len���� �۰ų� ����.
	 *
	 *	serial ����� ���� ���ŵǴ� message�� ���ڷ� ������ pointer�� ��ü�� �����Ѵ�.
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
	 *	@brief  serial ����� �۽� buffer�� ���.
	 *
	 *	���۵��� ���� �۽� buffer�� ��� message�� �����Ѵ�.
	 *	������ �۽� message�� ������ �������� �� �� �����Ƿ� ����ġ ���� error�� �߻���ų �� �ִ�.
	 */
	void clearSendBuffer();
	/**
	 *	@brief  serial ����� ���� buffer�� ���.
	 *
	 *	serial ��� channel�� ���ŵ� ���� buffer�� ��� message�� �����Ѵ�.
	 *	������ ���� message�� ������ �������� �� �� �����Ƿ� ����ġ ���� error�� �߻���ų �� �ִ�.
	 */
	void clearReceiveBuffer();

	/**
	 *	@brief  serial ����� �۽� buffer�� ��� �ִ����� Ȯ��.
	 *	@return  �۽� buffer�� ��� �ִٸ� true�� ��ȯ.
	 *
	 *	serial ����� ���� ������ message�� �۽� buffer�� ��� �ִ��� ���θ� ��ȯ�Ѵ�.
	 */
	bool isSendBufferEmpty() const;
	/**
	 *	@brief  serial ����� ���� buffer�� ��� �ִ����� Ȯ��.
	 *	@return  ���� buffer�� ��� �ִٸ� true�� ��ȯ.
	 *
	 *	serial ����� ���� ���ŵ� message�� ���� buffer�� ��� �ִ��� ���θ� ��ȯ�Ѵ�.
	 */
	bool isReceiveBufferEmpty() const;

	/**
	 *	@brief  serial ����� ���� �۽��� message�� ���̸� ��ȯ.
	 *	@return  �۽� message�� ���̸� ��ȯ.
	 *
	 *	serial ����� ���� ������ message�� �����ϰ� �ִ� �۽� buffer�� ���̸� ��ȯ�Ѵ�.
	 */
	std::size_t getSendBufferSize() const;
	/**
	 *	@brief  serial ����� ���� ���ŵ� message�� ���̸� ��ȯ.
	 *	@return  ���ŵ� message�� ���̸� ��ȯ.
	 *
	 *	serial ����� ���� ���ŵ� message�� �����ϰ� �ִ� ���� buffer�� ���̸� ��ȯ�Ѵ�.
	 */
	std::size_t getReceiveBufferSize() const;

protected:
	/**
	 *	@brief  �۽� buffer�� ����� message�� ������ ����.
	 *
	 *	�۽� buffer�� ����Ǿ� �ִ� message�� asynchronous I/O�� ���� �۽��Ѵ�.
	 */
	virtual void doStartSending();
	/**
	 *	@brief  �۽� ��û�� message�� ������ �Ϸ�� ��� ȣ��Ǵ� completion routine.
	 *	@param[in]  ec  message�� �����ϴ� �������� �߻��� ������ error code.
	 *	@throw  LogException  serial port�� close �������� error�� �߻�.
	 *
	 *	asynchronous I/O�� �̿��Ͽ� �۽� ��û�� message�� ������ �Ϸ�Ǿ��� �� system�� ���� ȣ��Ǵ� completion routine�̴�.
	 *	doStartSending() �Լ� ������ asynchronous �۽� ��û�� �ϸ鼭 �ش� �Լ��� completion routine���� ������ �־�� �Ѵ�.
	 */
	virtual void doCompleteSending(const boost::system::error_code &ec);
	/**
	 *	@brief  serial ��� channel�� ���� ������ message�� receive buffer�� ���� ����.
	 *
	 *	serial ����� ���ŵǴ� message�� asynchronous I/O�� �̿��Ͽ� �����ϱ� �����Ѵ�.
	 */
	virtual void doStartReceiving();
	/**
	 *	@brief  serial ��� channel�� ���� ���ŵ� message�� �ִ� ��� ȣ��Ǵ� completion routine.
	 *	@param[in]  ec  message�� �����ϴ� �������� �߻��� ������ error code.
	 *	@param[in]  bytesTransferred  ���ŵ� message�� ����.
	 *	@throw  LogException  serial port�� close �������� error�� �߻�.
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
	 *	@brief  serial ����� ���������� �����ϴ� Boost.ASIO�� serial port ��ü.
	 */
	boost::asio::serial_port port_;

	/**
	 *	@brief  serial ��� channel�� ����Ǿ� �ְ� ���� ���������� Ȯ���ϴ� flag ����.
	 *
	 *	serial ��� channel�� ����Ǿ� �ְ� ���� ���¶�� true��, �׷��� �ʴٸ� false�� ǥ���Ѵ�.
	 */
	bool isActive_;

	/**
	 *	@brief  serial ����� ���� send buffer.
	 *
	 *	GuardedByteBuffer�� ��ü�� multi-thread ȯ�濡���� �����ϰ� ����� �� �ִ�.
	 */
	GuardedByteBuffer sendBuffer_;
	/**
	 *	@brief  serial ����� ���� send buffer.
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


#endif  // __SWL_UTIL__SERIAL_PORT__H_
