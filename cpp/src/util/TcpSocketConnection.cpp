#include "swl/Config.h"
#include "swl/util/TcpSocketConnection.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


#if defined(min)
#undef min
#endif


namespace swl {

///*static*/ TcpSocketConnection::pointer TcpSocketConnection::create(boost::asio::io_service &ioService)
//{
//	return pointer(new TcpSocketConnection(ioService));
//}

TcpSocketConnection::TcpSocketConnection(boost::asio::io_service &ioService)
: socket_(ioService),
  receiveBuffer_(), sendBuffer_(), sentMsgLength_(0),
  isReceiving_(false), isSending_(false)
{}

TcpSocketConnection::~TcpSocketConnection()
{
}

void TcpSocketConnection::start()
{
	// put the socket into non-blocking mode.
	boost::asio::ip::tcp::socket::non_blocking_io non_blocking_io(true);
	socket_.io_control(non_blocking_io);

	doStartOperation();
}

void TcpSocketConnection::doStartOperation()
{
	// start a read operation if the third party library wants one.
	if (!isReceiving_)
	{
		isReceiving_ = true;
		socket_.async_read_some(
			boost::asio::buffer(receiveMsg_),
			// caution: shared_from_this() must be used here
			boost::bind(&TcpSocketConnection::doCompleteReceiving, shared_from_this(), boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred)
		);
	}

	// start a write operation if the third party library wants one.
	if (!isSending_ && !sendBuffer_.isEmpty())
	{
		isSending_ = true;
#if defined(__GNUC__)
		sentMsgLength_ = std::min(sendBuffer_.getSize(), (std::size_t)MAX_SEND_LENGTH_);
#else
		sentMsgLength_ = std::min(sendBuffer_.getSize(), MAX_SEND_LENGTH_);
#endif
		sendBuffer_.top(sendMsg_.c_array(), sentMsgLength_);
		boost::asio::async_write(
			socket_,
			boost::asio::buffer(sendMsg_, sentMsgLength_),
			// caution: shared_from_this() must be used here
			boost::bind(&TcpSocketConnection::doCompleteSending, shared_from_this(), boost::asio::placeholders::error)
		);
	}
}

void TcpSocketConnection::doCompleteSending(boost::system::error_code ec)
{
	isSending_ = false;

	// notify third party library that it can perform a write.
	if (!ec)
	{
		// TODO [add] >> do something here to process sending message

		sendBuffer_.pop(sentMsgLength_);
		sentMsgLength_ = 0;
	}

	// the third party library successfully performed a write on the socket.
	// start new read or write operations based on what it now wants.
	if (!ec || boost::asio::error::would_block == ec)
		doStartOperation();
	// otherwise, an error occurred. Closing the socket cancels any outstanding asynchronous read or write operations.
	// the TcpSocketConnection object will be destroyed automatically once those outstanding operations complete.
	else
		socket_.close();
}

void TcpSocketConnection::doCompleteReceiving(boost::system::error_code ec, std::size_t bytesTransferred)
{
	isReceiving_ = false;

	// notify third party library that it can perform a read.
	if (!ec)
	{
		// TODO [add] >> do something here to process received message

		receiveBuffer_.push(receiveMsg_.data(), bytesTransferred);
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

void TcpSocketConnection::clearSendBuffer()
{
	sendBuffer_.clear();
}

void TcpSocketConnection::clearReceiveBuffer()
{
	receiveBuffer_.clear();
}

bool TcpSocketConnection::isSendBufferEmpty() const
{
	return sendBuffer_.isEmpty();
}

bool TcpSocketConnection::isReceiveBufferEmpty() const
{
	return receiveBuffer_.isEmpty();
}

std::size_t TcpSocketConnection::getSendBufferSize() const
{
	return sendBuffer_.getSize();
}

std::size_t TcpSocketConnection::getReceiveBufferSize() const
{
	return receiveBuffer_.getSize();
}

}  // namespace swl
