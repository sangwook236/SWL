#include "swl/Config.h"
#include "swl/util/AsyncTcpSocketClient.h"
#include "swl/base/LogException.h"
#include "swl/base/String.h"
#include <boost/bind.hpp>


#if defined(_DEBUG)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


#if defined(min)
#undef min
#endif


namespace swl {

//-----------------------------------------------------------------------------------
//

#if defined(_UNICODE) || defined(UNICODE)
AsyncTcpSocketClient::AsyncTcpSocketClient(boost::asio::io_service &ioService, const std::wstring &hostName, const std::wstring &serviceName)
#else
AsyncTcpSocketClient::AsyncTcpSocketClient(boost::asio::io_service &ioService, const std::string &hostName, const std::string &serviceName)
#endif
: socket_(ioService), isActive_(false),
  receiveBuffer_(), sendBuffer_(), sentMsgLength_(0)
{
	boost::asio::ip::tcp::resolver resolver(socket_.get_io_service());
#if defined(_UNICODE) || defined(UNICODE)
	boost::asio::ip::tcp::resolver::query query(String::wcs2mbs(hostName), String::wcs2mbs(serviceName));
#else
	boost::asio::ip::tcp::resolver::query query(hostName, serviceName);
#endif
	boost::asio::ip::tcp::resolver::iterator endpoint_iterator = resolver.resolve(query);

	socket_.get_io_service().post(boost::bind(&AsyncTcpSocketClient::doStartConnecting, this, endpoint_iterator));
}

AsyncTcpSocketClient::~AsyncTcpSocketClient()
{
	if (isActive_)
		disconnect();
}

void AsyncTcpSocketClient::disconnect()
{
	socket_.get_io_service().post(boost::bind(&AsyncTcpSocketClient::doCloseOperation, this, boost::system::error_code()));
}

void AsyncTcpSocketClient::send(const unsigned char *msg, const std::size_t len)
{
	socket_.get_io_service().post(boost::bind(&AsyncTcpSocketClient::doSendOperation, this, msg, len));
}

std::size_t AsyncTcpSocketClient::receive(unsigned char *msg, const std::size_t len)
{
	if (receiveBuffer_.isEmpty()) return 0;

	const std::size_t readLen = std::min(len, receiveBuffer_.getSize());
	receiveBuffer_.top(msg, readLen);
	receiveBuffer_.pop(readLen);
	return readLen;
}

void AsyncTcpSocketClient::cancelIo()
{
	socket_.get_io_service().post(boost::bind(&AsyncTcpSocketClient::doCancelOperation, this, boost::system::error_code()));
}

void AsyncTcpSocketClient::clearSendBuffer()
{
	sendBuffer_.clear();
}

void AsyncTcpSocketClient::clearReceiveBuffer()
{
	receiveBuffer_.clear();
}

bool AsyncTcpSocketClient::isSendBufferEmpty() const
{
	return sendBuffer_.isEmpty();
}

bool AsyncTcpSocketClient::isReceiveBufferEmpty() const
{
	return receiveBuffer_.isEmpty();
}

std::size_t AsyncTcpSocketClient::getSendBufferSize() const
{
	return sendBuffer_.getSize();
}

std::size_t AsyncTcpSocketClient::getReceiveBufferSize() const
{
	return receiveBuffer_.getSize();
}

void AsyncTcpSocketClient::doStartConnecting(boost::asio::ip::tcp::resolver::iterator endpoint_iterator)
{
	// asynchronously connect a socket to the specified remote endpoint and call doCompleteConnecting() when it completes or fails
	socket_.async_connect(
		*endpoint_iterator,
		boost::bind(&AsyncTcpSocketClient::doCompleteConnecting, this, boost::asio::placeholders::error, ++endpoint_iterator)
	);
}

void AsyncTcpSocketClient::doCompleteConnecting(const boost::system::error_code &ec, boost::asio::ip::tcp::resolver::iterator endpoint_iterator)
{
	// the connection to the server has now completed or failed and returned an error 
	if (!ec)  // success, so start waiting for read data 
	{
		//boost::asio::ip::tcp::socket::non_blocking_io non_blocking_io(true);
		//socket_.io_control(non_blocking_io);

		isActive_ = true;

		doStartReceiving();
		// TODO [check] >>
		//if (!sendBuffer_.isEmpty()) doStartSending();
	}
	else if (endpoint_iterator != boost::asio::ip::tcp::resolver::iterator())  // boost::asio::ip::tcp::resolver::iterator end;
	{
		// failed, so wait for another connection event
		socket_.close();
		doStartConnecting(endpoint_iterator);
	}
}

void AsyncTcpSocketClient::doStartSending()
{
	sentMsgLength_ = std::min(sendBuffer_.getSize(), MAX_SEND_LENGTH_);
	sendBuffer_.top(sendMsg_.c_array(), sentMsgLength_);
	boost::asio::async_write(
		socket_,
		boost::asio::buffer(sendMsg_, sentMsgLength_),
		boost::bind(&AsyncTcpSocketClient::doCompleteSending, this, boost::asio::placeholders::error)
	);
}

void AsyncTcpSocketClient::doCompleteSending(const boost::system::error_code &ec)
{
	if (!ec)
	{
		// TODO [add] >> do something here to process sending message

		// FIXME [check] >> bytes transferred == len ???
		sendBuffer_.pop(sentMsgLength_);
		sentMsgLength_ = 0;
		if (!sendBuffer_.isEmpty())
			doStartSending();
	}
	else
		doCloseOperation(ec);
}

void AsyncTcpSocketClient::doStartReceiving()
{
	socket_.async_read_some(
		boost::asio::buffer(receiveMsg_),
		boost::bind(&AsyncTcpSocketClient::doCompleteReceiving, this, boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred)
	);
} 

void AsyncTcpSocketClient::doCompleteReceiving(const boost::system::error_code &ec, std::size_t bytesTransferred)
{
	if (!ec)
	{
		// TODO [add] >> do something here to process receiving message

		receiveBuffer_.push(receiveMsg_.data(), bytesTransferred);
		doStartReceiving();
	}
	else
		doCloseOperation(ec);
}

void AsyncTcpSocketClient::doSendOperation(const unsigned char *msg, const std::size_t len)
{
	const bool write_in_progress = !sendBuffer_.isEmpty();
	sendBuffer_.push(msg, msg + len);
	if (!write_in_progress)
		doStartSending();
}

void AsyncTcpSocketClient::doCloseOperation(const boost::system::error_code &ec)
{
	// if this call is the result of a timer cancel()
	if (boost::asio::error::operation_aborted == ec)
		return;

	if (boost::asio::error::eof == ec)
	{
		// connection closed cleanly by peer.
	}
	else if (ec)  // some other error.
	{
		//throw boost::system::system_error(ec);
		throw std::runtime_error(ec.message());
	}

	socket_.close();
	isActive_ = false;
}

void AsyncTcpSocketClient::doCancelOperation(const boost::system::error_code &ec)
{
	if (boost::asio::error::eof == ec)
	{
		// connection closed cleanly by peer.
	}
	else if (ec)  // some other error.
	{
		//throw boost::system::system_error(ec);
		throw LogException(LogException::L_ERROR, ec.message(), __FILE__, __LINE__, __FUNCTION__);
	}

	socket_.cancel();
}

}  // namespace swl
