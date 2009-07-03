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
 *	@brief  동기적으로 TCP socket 통신을 수행하는 client class.
 *
 *	TCP socket 통신을 통해 message를 송수신하기 위해 send() 함수와 receive() 함수를 호출하면 된다.
 *	TCP socket 통신을 수행하는 개략적인 절차는 아래와 같다.
 *		- TcpSocketClient 객체 생성
 *		- connect() 함수를 이용하여 TCP server와 연결
 *		- send() and/or receive() 함수를 이용해 message 송수신
 *		- 작업이 끝났다면, disconnect() 함수를 호출하여 연결 해제
 *		- TcpSocketClient 객체 소멸
 *
 *	synchronous I/O를 사용하여 송수신을 수행한다.
 */
class SWL_UTIL_API TcpSocketClient
{
public:
	//typedef TcpSocketClient base_type;

public:
	/**
	 *	@brief  [ctor] contructor.
	 *	@param[in]  ioService  TCP socket 통신을 위한 Boost.ASIO의 I/O service 객체.
	 *
	 *	TCP socket 통신을 위해 필요한 설정들을 초기화한다.
	 */
	TcpSocketClient(boost::asio::io_service &ioService);
	/**
	 *	@brief  [dtor] default destructor.
	 *
	 *	TCP socket 통신을 종료하기 위한 절차를 수행한다.
	 *	통신 channel이 열려 있는 경우 disconnect() 함수를 호출하여 이를 닫는다.
	 */
	~TcpSocketClient();

public:
	/**
	 *	@brief  지정된 host 이름과 service 이름을 이용해 TCP socket server와 channel을 연결.
	 *	@param[in]  hostName  TCP socket server의 host 이름.
	 *	@param[in]  serviceName  TCP socket server의 service 이름.
	 *	@return  TCP socket channel이 정상적으로 연결되었다면 true 반환.
	 *
	 *	인자로 넘겨진 host 이름과 service 이름을 이용하여 TCP socket channel을 연결하고
	 *
	 *	host 이름은 IP address이나 domain 이름으로 지정할 수 있다.
	 *		- "abc.com"
	 *		- "100.110.120.130"
	 *	service 이름은 이나 port 번호로 지정할 수 있다.
	 *		- "http" or "daytime"
	 *		- "80"
	 */
#if defined(_UNICODE) || defined(UNICODE)
	bool connect(const std::wstring &hostName, const std::wstring &serviceName);
#else
	bool connect(const std::string &hostName, const std::string &serviceName);
#endif
	/**
	 *	@brief  TCP socket 통신 channel의 연결을 해제.
	 *
	 *	TCP socket 통신 channel의 연결을 끊고, 사용 중인 resource를 반환한다.
	 */
	void disconnect();

	/**
	 *	@brief  TCP socket 통신이 연결 상태에 있는지 확인.
	 *	@return  TCP socket 통신 channel이 연결 상태이면 true 반환.
	 *
	 *	TCP socket 통신 channel의 연결 상태를 반환한다.
	 */
	bool isConnected() const  {  return isActive_;  }

	/**
	 *	@brief  지정된 message를 연결된 TCP socket 통신 channel을 통해 전송.
	 *	@param[in]  msg  전송할 message를 지정하는 pointer.
	 *	@param[in]  len  전송할 message 길이.
	 *	@throw  std::runtime_error  송신 operation 동안 error가 발생.
	 *	@return  실제로 송신된 message의 길이를 반환. 인자로 지정된 len보다 작거나 같음.
	 *
	 *	요청된 message를 TCP socket 통신을 통해 전송한다.
	 *	송신된 message의 길이는 인자로 주어진 길이보다 작거나 같다
	 *	synchronous I/O를 통해 message를 전송한다.
	 */
	size_t send(const unsigned char *msg, const size_t len);
	/**
	 *	@brief  연결된 TCP socket 통신 channel을 통해 message를 수신.
	 *	@param[out]  msg  수신된 message를 저장할 pointer.
	 *	@param[in]  len  synchronous I/O를 통해 수신한 message를 저장할 buffer의 크기를 지정.
	 *	@throw  std::runtime_error  수신 operation 동안 error가 발생.
	 *	@return  실제로 수신된 message의 길이를 반환. 인자로 지정된 len보다 작거나 같음.
	 *
	 *	TCP socket 통신을 통해 수신되는 message를 인자로 지정된 pointer의 객체에 저장한다.
	 *	수신된 message의 길이는 인자로 주어진 길이보다 작거나 같다
	 *	synchronous I/O를 통해 message를 수신한다.
	 */
	size_t receive(unsigned char *msg, const size_t len);

private:
	boost::asio::ip::tcp::socket socket_;

	bool isActive_;
};

//-----------------------------------------------------------------------------------
//

/**
 *	@brief  비동기적으로 TCP socket 통신을 수행하는 client class.
 *
 *	TCP socket 통신을 통해 message를 송수신하기 위해 send() 함수와 receive() 함수를 호출하면 된다.
 *	TCP socket 통신을 수행하는 개략적인 절차는 아래와 같다.
 *		- AsyncTcpSocketClient 객체 생성
 *		- connect() 함수를 이용하여 TCP server와 연결
 *		- send() and/or receive() 함수를 이용해 message 송수신
 *		- 작업이 끝났다면, disconnect() 함수를 호출하여 연결 해제
 *		- AsyncTcpSocketClient 객체 소멸
 *
 *	asynchronous I/O를 사용하여 송수신을 수행한다.
 */
class SWL_UTIL_API AsyncTcpSocketClient
{
public:
	//typedef AsyncTcpSocketClient base_type;

public:
	/**
	 *	@brief  [ctor] contructor.
	 *	@param[in]  ioService  TCP socket 통신을 위한 Boost.ASIO의 I/O service 객체.
	 *
	 *	TCP socket 통신을 위해 필요한 설정들을 초기화한다.
	 */
	AsyncTcpSocketClient(boost::asio::io_service &ioService);
	/**
	 *	@brief  [dtor] default destructor.
	 *
	 *	TCP socket 통신을 종료하기 위한 절차를 수행한다.
	 *	통신 channel이 열려 있는 경우 disconnect() 함수를 호출하여 이를 닫는다.
	 */
	~AsyncTcpSocketClient();

public:
	/**
	 *	@brief  지정된 host 이름과 service 이름을 이용해 TCP socket server와 channel을 연결.
	 *	@param[in]  hostName  TCP socket server의 host 이름.
	 *	@param[in]  serviceName  TCP socket server의 service 이름.
	 *	@return  TCP socket channel이 정상적으로 연결되었다면 true 반환.
	 *
	 *	인자로 넘겨진 host 이름과 service 이름을 이용하여 TCP socket channel을 연결하고
	 *
	 *	host 이름은 IP address이나 domain 이름으로 지정할 수 있다.
	 *		- "abc.com"
	 *		- "100.110.120.130"
	 *	service 이름은 이나 port 번호로 지정할 수 있다.
	 *		- "http" or "daytime"
	 *		- "80"
	 */
#if defined(_UNICODE) || defined(UNICODE)
	bool connect(const std::wstring &hostName, const std::wstring &serviceName);
#else
	bool connect(const std::string &hostName, const std::string &serviceName);
#endif
	/**
	 *	@brief  TCP socket 통신 channel의 연결을 해제.
	 *	@throw  std::runtime_error  TCP socket의 close 과정에서 error가 발생.
	 *
	 *	TCP socket 통신 channel의 연결을 끊고, 사용 중인 resource를 반환한다.
	 */
	void disconnect();

	/**
	 *	@brief  TCP socket 통신이 연결 상태에 있는지 확인.
	 *	@return  TCP socket 통신 channel이 연결 상태이면 true 반환.
	 *
	 *	TCP socket 통신 channel의 연결 상태를 반환한다.
	 */
	bool isConnected() const  {  return isActive_;  }

	/**
	 *	@brief  지정된 message를 연결된 TCP socket 통신 channel을 통해 전송.
	 *	@param[in]  msg  전송할 message를 지정하는 pointer.
	 *	@param[in]  len  전송할 message 길이.
	 *	@return  실제로 송신된 message의 길이를 반환. 인자로 지정된 len보다 작거나 같음.
	 *
	 *	요청된 message를 TCP socket 통신을 통해 전송한다.
	 *	송신된 message의 길이는 인자로 주어진 길이보다 작거나 같다
	 *	asynchronous I/O를 통해 message를 전송한다.
	 */
	void send(const unsigned char *msg, const size_t len);
	/**
	 *	@brief  연결된 TCP socket 통신 channel을 통해 message를 수신.
	 *	@param[out]  msg  수신된 message를 저장할 pointer.
	 *	@param[in]  len  synchronous I/O를 통해 수신한 message를 저장할 buffer의 크기를 지정.
	 *	@return  실제로 수신된 message의 길이를 반환. 인자로 지정된 len보다 작거나 같음.
	 *
	 *	TCP socket 통신을 통해 수신되는 message를 인자로 지정된 pointer의 객체에 저장한다.
	 *	수신된 message의 길이는 인자로 주어진 길이보다 작거나 같다
	 *	asynchronous I/O를 통해 message를 수신한다.
	 */
	void receive(const size_t len);

	/**
	 *	@brief  수신된 message를 반환.
	 *	@param[out]  buf  message 수신 요청 후 수신된 message를 저장한 buffer.
	 *
	 *	message 수신 요청 후 server로부터 수신된 message의 내용을 가져온다.
	 *	buf 안에 있는 기존 data는 모두 삭제된다.
	 */
	void getReceiveMessage(std::vector<unsigned char> &buf) const;

	/**
	 *	@brief  수행 중인 I/O 작업을 취소.
	 *	@throw  std::runtime_error  송수신 operation을 취소하는 과정에서 error가 발생.
	 *
	 *	asynchronous I/O를 통해 진행 중인 송수신 operation을 취소한다.
	 */
	void cancelIo();

	/**
	 *	@brief  server 접속 요청이 완료되었는지 확인.
	 *	@return  요청한 server 접속이 완료되었다면 true를 반환.
	 *
	 *	server 접속 요청이 정상적으로 완료되었는지를 확인한다.
	 */
	bool isConnectDone() const  {  return isConnectDone_;  }
	/**
	 *	@brief  message 송신이 완료되었는지 확인.
	 *	@return  요청한 message 송신이 완료되었다면 true를 반환.
	 *
	 *	message 송신 요청이 정상적으로 완료되었는지를 확인한다.
	 */
	bool isSendDone() const  {  return isSendDone_;  }
	/**
	 *	@brief  message 수신이 완료되었는지 확인.
	 *	@return  요청한 message 수신이 완료되었다면 true를 반환.
	 *
	 *	message 수신 요청이 정상적으로 완료되었는지를 확인한다.
	 */
	bool isReceiveDone() const  {  return isReceiveDone_;  }

private:
	void completeConnecting(const boost::system::error_code &ec);
	void completeSending(const boost::system::error_code &ec);
	void completeReceiving(const boost::system::error_code &ec, size_t bytesTransferred);

	void doCloseOperation(const boost::system::error_code &ec);
	void doCancelOperation(const boost::system::error_code &ec);

private:
	boost::asio::ip::tcp::socket socket_;

	bool isActive_;

	std::vector<unsigned char> sendMsg_;
	std::vector<unsigned char> receiveMsg_;

	bool isConnectDone_;
	bool isSendDone_;
	bool isReceiveDone_;
};

}  // namespace swl


#endif  // __SWL_UTIL__TCP_SOCKET_CLIENT__H_
