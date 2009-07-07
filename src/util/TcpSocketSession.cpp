#include "swl/Config.h"
#include "swl/util/TcpSocketSession.h"


#if defined(_DEBUG)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


#if defined(min)
#undef min
#endif


namespace swl {

TcpSocketSession::TcpSocketSession(boost::asio::ip::tcp::socket &socket)
: socket_(socket), state_(RECEIVING),
  receiveBuffer_(), sendBuffer_()
{
}

TcpSocketSession::~TcpSocketSession()
{
}

void TcpSocketSession::send(boost::system::error_code &ec)
{
	if (sendBuffer_.isEmpty())
		state_ = RECEIVING;
	else
	{
		const std::size_t len = std::min(sendBuffer_.getSize(), MAX_SEND_LENGTH_);
		sendBuffer_.top(sendMsg_.c_array(), len);
		if (const std::size_t sentLen = socket_.write_some(boost::asio::buffer(sendMsg_, len), ec))
		{
			// TODO [add] >> do something here to process sending message

			sendBuffer_.pop(sentLen);
			state_ = sendBuffer_.isEmpty() ? RECEIVING : SENDING;
		}
	}
}

void TcpSocketSession::receive(boost::system::error_code &ec)
{
	if (const std::size_t len = socket_.read_some(boost::asio::buffer(receiveMsg_), ec))
	{
		// TODO [add] >> do something here to process received message

		receiveBuffer_.push(receiveMsg_.data(), len);
		state_ = sendBuffer_.isEmpty() ? RECEIVING : SENDING;
	}
}

void TcpSocketSession::clearSendBuffer()
{
	sendBuffer_.clear();
}

void TcpSocketSession::clearReceiveBuffer()
{
	receiveBuffer_.clear();
}

bool TcpSocketSession::isSendBufferEmpty() const
{
	return sendBuffer_.isEmpty();
}

bool TcpSocketSession::isReceiveBufferEmpty() const
{
	return receiveBuffer_.isEmpty();
}

std::size_t TcpSocketSession::getSendBufferSize() const
{
	return sendBuffer_.getSize();
}

std::size_t TcpSocketSession::getReceiveBufferSize() const
{
	return receiveBuffer_.getSize();
}

}  // namespace swl
