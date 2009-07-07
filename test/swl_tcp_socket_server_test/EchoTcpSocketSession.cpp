#include "swl/Config.h"
#include "EchoTcpSocketSession.h"


#if defined(_DEBUG)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


#if defined(min)
#undef min
#endif


namespace swl {

EchoTcpSocketSession::EchoTcpSocketSession(boost::asio::ip::tcp::socket &socket)
: base_type(socket)
{
}

EchoTcpSocketSession::~EchoTcpSocketSession()
{
}

void EchoTcpSocketSession::send(boost::system::error_code &ec)
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
			std::cout << "\tsend>>>>> ";
			std::cout.write((char *)sendMsg_.data(), (std::streamsize)sentLen);
			std::cout << std::endl;

			sendBuffer_.pop(sentLen);
			state_ = sendBuffer_.isEmpty() ? RECEIVING : SENDING;
		}
	}
}

void EchoTcpSocketSession::receive(boost::system::error_code &ec)
{
	if (const std::size_t len = socket_.read_some(boost::asio::buffer(receiveMsg_), ec))
	{
		// TODO [add] >> do something here to process received message
		std::cout << "\treceive<<<<< ";
		std::cout.write((char *)receiveMsg_.data(), (std::streamsize)len);
		std::cout << std::endl;

		//receiveBuffer_.push(receiveMsg_.data(), len);
		sendBuffer_.push(receiveMsg_.data(), len);
		state_ = sendBuffer_.isEmpty() ? RECEIVING : SENDING;
	}
}

}  // namespace swl
