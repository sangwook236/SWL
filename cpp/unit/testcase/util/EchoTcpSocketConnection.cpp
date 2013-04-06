#include "swl/Config.h"
#include "EchoTcpSocketConnection.h"
#include <iostream>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


#if defined(min)
#undef min
#endif


namespace swl {

/*static*/ EchoTcpSocketConnection::pointer EchoTcpSocketConnection::create(boost::asio::io_service &ioService)
{
	return pointer(new EchoTcpSocketConnection(ioService));
}

EchoTcpSocketConnection::EchoTcpSocketConnection(boost::asio::io_service &ioService)
: base_type(ioService)
{}

void EchoTcpSocketConnection::doStartOperation()
{
	base_type::doStartOperation();
}

void EchoTcpSocketConnection::doCompleteSending(boost::system::error_code ec)
{
	isSending_ = false;

	// notify third party library that it can perform a write.
	if (!ec)
	{
		// TODO [add] >> do something here to process sending message
		std::cout << "\tsend>>>>> ";
		std::cout.write((char *)sendMsg_.data(), (std::streamsize)sentMsgLength_);
		std::cout << std::endl;

		sendBuffer_.pop(sentMsgLength_);
		sentMsgLength_ = 0;
	}

	// the third party library successfully performed a write on the socket.
	// start new read or write operations based on what it now wants.
	if (!ec || boost::asio::error::would_block == ec)
		doStartOperation();
	// otherwise, an error occurred. Closing the socket cancels any outstanding asynchronous read or write operations.
	// the EchoTcpSocketConnection object will be destroyed automatically once those outstanding operations complete.
	else
		socket_.close();
}

void EchoTcpSocketConnection::doCompleteReceiving(boost::system::error_code ec, std::size_t bytesTransferred)
{
	isReceiving_ = false;

	// notify third party library that it can perform a read.
	if (!ec)
	{
		// TODO [add] >> do something here to process received message
		std::cout << "\treceive<<<<< ";
		std::cout.write((char *)receiveMsg_.data(), (std::streamsize)bytesTransferred);
		std::cout << std::endl;

		//receiveBuffer_.push(receiveMsg_.data(), bytesTransferred);
		sendBuffer_.push(receiveMsg_.data(), bytesTransferred);
	}

	// the third party library successfully performed a read on the socket.
	// start new read or write operations based on what it now wants.
	if (!ec || boost::asio::error::would_block == ec)
		doStartOperation();
	// otherwise, an error occurred. Closing the socket cancels any outstanding asynchronous read or write operations.
	// the TcpSocketConnection_FullDuplex object will be destroyed automatically once those outstanding operations complete.
	else
		socket_.close();
}

}  // namespace swl
