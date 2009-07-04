#include "swl/Config.h"
#include "swl/util/FullDuplexTcpSocketSession.h"
#include "swl/base/LogException.h"
#include <boost/asio.hpp>
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

FullDuplexTcpSocketSession::FullDuplexTcpSocketSession(boost::asio::ip::tcp::socket &socket)
: socket_(socket), isActive_(false),
  receiveBuffer_(), sendBuffer_(), sentMsgLength_(0)
{
}

FullDuplexTcpSocketSession::~FullDuplexTcpSocketSession()
{
	if (isActive())
		close();
}

void FullDuplexTcpSocketSession::close()
{
	socket_.get_io_service().post(boost::bind(&FullDuplexTcpSocketSession::doCloseOperation, this, boost::system::error_code()));
}

void FullDuplexTcpSocketSession::send(const unsigned char *msg, const size_t len)
{
	socket_.get_io_service().post(boost::bind(&FullDuplexTcpSocketSession::doSendOperation, this, msg, len));
}

size_t FullDuplexTcpSocketSession::receive(unsigned char *msg, const size_t len)
{
	if (receiveBuffer_.isEmpty()) return 0;

	const size_t sz = std::min(len, receiveBuffer_.getSize());
	receiveBuffer_.top(msg, sz);
	receiveBuffer_.pop(sz);
	return sz;
}

void FullDuplexTcpSocketSession::cancelIo()
{
	socket_.get_io_service().post(boost::bind(&FullDuplexTcpSocketSession::doCancelOperation, this, boost::system::error_code()));
}

void FullDuplexTcpSocketSession::clearSendBuffer()
{
	sendBuffer_.clear();
}

void FullDuplexTcpSocketSession::clearReceiveBuffer()
{
	receiveBuffer_.clear();
}

bool FullDuplexTcpSocketSession::isSendBufferEmpty() const
{
	return sendBuffer_.isEmpty();
}

bool FullDuplexTcpSocketSession::isReceiveBufferEmpty() const
{
	return receiveBuffer_.isEmpty();
}

size_t FullDuplexTcpSocketSession::getSendBufferSize() const
{
	return sendBuffer_.getSize();
}

size_t FullDuplexTcpSocketSession::getReceiveBufferSize() const
{
	return receiveBuffer_.getSize();
}

void FullDuplexTcpSocketSession::doSendOperation(const unsigned char *msg, const size_t len)
{
	const bool write_in_progress = !sendBuffer_.isEmpty();
	sendBuffer_.push(msg, msg + len);
	if (!write_in_progress)
		startSending();
}

void FullDuplexTcpSocketSession::startSending()
{
	if (!sendBuffer_.isEmpty())
	{
		sentMsgLength_ = std::min(sendBuffer_.getSize(), MAX_SEND_LENGTH_);
		sendBuffer_.top(sendMsg_, sentMsgLength_);
		boost::asio::async_write(
			socket_,
			boost::asio::buffer(sendMsg_, sentMsgLength_),
			boost::bind(&FullDuplexTcpSocketSession::completeSending, this, boost::asio::placeholders::error)
		);
	}
}

void FullDuplexTcpSocketSession::completeSending(const boost::system::error_code &ec)
{
	if (!ec)
	{
		sendBuffer_.pop(sentMsgLength_);
		sentMsgLength_ = 0;
		startSending();
	}
	else
		doCloseOperation(ec);
}

void FullDuplexTcpSocketSession::startReceiving()
{
	socket_.async_read_some(
		boost::asio::buffer(receiveMsg_, MAX_RECEIVE_LENGTH_),
		boost::bind(&FullDuplexTcpSocketSession::completeReceiving, this, boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred)
	); 
} 

void FullDuplexTcpSocketSession::completeReceiving(const boost::system::error_code &ec, size_t bytesTransferred)
{
	if (!ec)
	{
		receiveBuffer_.push(receiveMsg_, bytesTransferred);
		startReceiving();
	}
	else
		doCloseOperation(ec);
}

void FullDuplexTcpSocketSession::doCloseOperation(const boost::system::error_code &ec)
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
		throw LogException(LogException::L_ERROR, ec.message(), __FILE__, __LINE__, __FUNCTION__);
	}

	socket_.close();
	isActive_ = false;
}

void FullDuplexTcpSocketSession::doCancelOperation(const boost::system::error_code &ec)
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
