#if !defined(__SWL_UTIL__TCP_SOCKET_CLIENT__H_)
#define __SWL_UTIL__TCP_SOCKET_CLIENT__H_ 1


#include "swl/util/ExportUtil.h"
#include <boost/asio.hpp>
#include <string>
#include <vector>


namespace swl {

//-----------------------------------------------------------------------------------
//

/**
 *	@brief  ���������� TCP socket ����� �����ϴ� client class.
 *
 *	TCP socket ����� ���� message�� �ۼ����ϱ� ���� send() �Լ��� receive() �Լ��� ȣ���ϸ� �ȴ�.
 *	TCP socket ����� �����ϴ� �������� ������ �Ʒ��� ����.
 *		- TcpSocketClient ��ü ����
 *		- connect() �Լ��� �̿��Ͽ� TCP server�� ����
 *		- send() and/or receive() �Լ��� �̿��� message �ۼ���
 *		- �۾��� �����ٸ�, disconnect() �Լ��� ȣ���Ͽ� ���� ����
 *		- TcpSocketClient ��ü �Ҹ�
 *
 *	synchronous I/O�� ����Ͽ� �ۼ����� �����Ѵ�.
 */
class SWL_UTIL_API TcpSocketClient
{
public:
	//typedef TcpSocketClient base_type;

public:
	/**
	 *	@brief  [ctor] constructor.
	 *	@param[in]  ioService  TCP socket ����� ���� Boost.ASIO�� I/O service ��ü.
	 *
	 *	TCP socket ����� ���� �ʿ��� �������� �ʱ�ȭ�Ѵ�.
	 */
	TcpSocketClient(boost::asio::io_service &ioService);
	/**
	 *	@brief  [dtor] virtual default destructor.
	 *
	 *	TCP socket ����� �����ϱ� ���� ������ �����Ѵ�.
	 *	��� channel�� ���� �ִ� ��� disconnect() �Լ��� ȣ���Ͽ� �̸� �ݴ´�.
	 */
	virtual ~TcpSocketClient();

public:
	/**
	 *	@brief  ������ host �̸��� service �̸��� �̿��� TCP socket server�� channel�� ����.
	 *	@param[in]  hostName  TCP socket server�� host �̸�.
	 *	@param[in]  serviceName  TCP socket server�� service �̸�.
	 *	@return  TCP socket channel�� ���������� ����Ǿ��ٸ� true ��ȯ.
	 *
	 *	���ڷ� �Ѱ��� host �̸��� service �̸��� �̿��Ͽ� TCP socket channel�� �����ϰ�
	 *
	 *	host �̸��� IP address�̳� domain �̸����� ������ �� �ִ�.
	 *		- "abc.com"
	 *		- "100.110.120.130"
	 *	service �̸��� �̳� port ��ȣ�� ������ �� �ִ�.
	 *		- "http" or "daytime"
	 *		- "80"
	 */
#if defined(_UNICODE) || defined(UNICODE)
	bool connect(const std::wstring &hostName, const std::wstring &serviceName);
#else
	bool connect(const std::string &hostName, const std::string &serviceName);
#endif
	/**
	 *	@brief  TCP socket ��� channel�� ������ ����.
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
	 *	@throw  LogException  �۽� operation ���� error�� �߻�.
	 *	@return  ������ �۽ŵ� message�� ���̸� ��ȯ. ���ڷ� ������ len���� �۰ų� ����.
	 *
	 *	��û�� message�� TCP socket ����� ���� �����Ѵ�.
	 *	�۽ŵ� message�� ���̴� ���ڷ� �־��� ���̺��� �۰ų� ����
	 *	synchronous I/O�� ���� message�� �����Ѵ�.
	 */
	virtual std::size_t send(const unsigned char *msg, const std::size_t len);
	/**
	 *	@brief  ����� TCP socket ��� channel�� ���� message�� ����.
	 *	@param[out]  msg  ���ŵ� message�� ������ pointer.
	 *	@param[in]  len  synchronous I/O�� ���� ������ message�� ������ buffer�� ũ�⸦ ����.
	 *	@throw  LogException  ���� operation ���� error�� �߻�.
	 *	@return  ������ ���ŵ� message�� ���̸� ��ȯ. ���ڷ� ������ len���� �۰ų� ����.
	 *
	 *	TCP socket ����� ���� ���ŵǴ� message�� ���ڷ� ������ pointer�� ��ü�� �����Ѵ�.
	 *	���ŵ� message�� ���̴� ���ڷ� �־��� ���̺��� �۰ų� ����
	 *	synchronous I/O�� ���� message�� �����Ѵ�.
	 */
	virtual std::size_t receive(unsigned char *msg, const std::size_t len);

protected:
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
};

}  // namespace swl


#endif  // __SWL_UTIL__TCP_SOCKET_CLIENT__H_
