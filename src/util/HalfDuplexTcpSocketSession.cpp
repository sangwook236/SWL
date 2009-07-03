#include "swl/util/HalfDuplexTcpSocketSession.h"
#include <boost/asio.hpp>


#if defined(_MSC_VER) && defined(_DEBUG)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


#if defined(min)
#undef min
#endif


namespace swl {

HalfDuplexTcpSocketSession::HalfDuplexTcpSocketSession(boost::asio::ip::tcp::socket &socket)
: socket_(socket), state_(RECEIVING),
  receiveBuffer_(), sendBuffer_(), sentMsgLength_(0)
{
}

HalfDuplexTcpSocketSession::~HalfDuplexTcpSocketSession()
{
}

void HalfDuplexTcpSocketSession::send(boost::system::error_code &ec)
{
	if (sendBuffer_.isEmpty())
		state_ = RECEIVING;
	else
	{
		sentMsgLength_ = std::min(sendBuffer_.getSize(), MAX_SEND_LENGTH_);
		sendBuffer_.top(sendMsg_, sentMsgLength_);
		if (const std::size_t len = socket_.write_some(boost::asio::buffer(sendMsg_, MAX_SEND_LENGTH_), ec))
		{
			sendBuffer_.pop(len);
			state_ = sendBuffer_.isEmpty() ? RECEIVING : SENDING;
		}
	}
}

void HalfDuplexTcpSocketSession::receive(boost::system::error_code &ec)
{
	if (const std::size_t len = socket_.read_some(boost::asio::buffer(receiveMsg_, MAX_RECEIVE_LENGTH_), ec))
	{
		receiveBuffer_.push(receiveMsg_, len);
		state_ = sendBuffer_.isEmpty() ? RECEIVING : SENDING;
	}
}

void HalfDuplexTcpSocketSession::clearSendBuffer()
{
	sendBuffer_.clear();
}

void HalfDuplexTcpSocketSession::clearReceiveBuffer()
{
	receiveBuffer_.clear();
}

bool HalfDuplexTcpSocketSession::isSendBufferEmpty() const
{
	return sendBuffer_.isEmpty();
}

bool HalfDuplexTcpSocketSession::isReceiveBufferEmpty() const
{
	return receiveBuffer_.isEmpty();
}

size_t HalfDuplexTcpSocketSession::getSendBufferSize() const
{
	return sendBuffer_.getSize();
}

size_t HalfDuplexTcpSocketSession::getReceiveBufferSize() const
{
	return receiveBuffer_.getSize();
}

}  // namespace swl
