#include "swl/Config.h"
#include "swl/util/TcpSocketClient.h"
#include "swl/base/LogException.h"
#include "swl/base/String.h"
#include <boost/bind.hpp>


#if defined(_DEBUG)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//-----------------------------------------------------------------------------------
//

TcpSocketClient::TcpSocketClient(boost::asio::io_service &ioService)
: socket_(ioService), isActive_(false)
{
}

TcpSocketClient::~TcpSocketClient()
{
	if (isActive_)
		disconnect();
}

#if defined(_UNICODE) || defined(UNICODE)
bool TcpSocketClient::connect(const std::wstring &hostName, const std::wstring &serviceName)
#else
bool TcpSocketClient::connect(const std::string &hostName, const std::string &serviceName)
#endif
{
	boost::asio::ip::tcp::resolver resolver(socket_.get_io_service());
#if defined(_UNICODE) || defined(UNICODE)
	boost::asio::ip::tcp::resolver::query query(String::wcs2mbs(hostName), String::wcs2mbs(serviceName));
#else
	boost::asio::ip::tcp::resolver::query query(hostName, serviceName);
#endif
	boost::asio::ip::tcp::resolver::iterator endpoint_iterator = resolver.resolve(query);
	boost::asio::ip::tcp::resolver::iterator end;

	boost::system::error_code ec = boost::asio::error::host_not_found;
	while (ec && endpoint_iterator != end)
	{
		socket_.close();
		socket_.connect(*endpoint_iterator++, ec);
	}
	if (ec) return false;

	isActive_ = true;
	return true;
}

void TcpSocketClient::disconnect()
{
	socket_.close();
	isActive_ = false;
}

std::size_t TcpSocketClient::send(const unsigned char *msg, const std::size_t len)
{
	boost::system::error_code ec;
	const std::size_t writtenLen = socket_.write_some(boost::asio::buffer(msg, len), ec);
	if (boost::asio::error::eof == ec)
	{
		// connection closed cleanly by peer.
		isActive_ = false;
	}
	else if (ec)  // some other error.
	{
		//throw boost::system::system_error(ec);
		throw LogException(LogException::L_ERROR, ec.message(), __FILE__, __LINE__, __FUNCTION__);
	}

	return writtenLen;
}

std::size_t TcpSocketClient::receive(unsigned char *msg, const std::size_t len)
{
	boost::system::error_code ec;
	const std::size_t readLen = socket_.read_some(boost::asio::buffer(msg, len), ec);
	if (boost::asio::error::eof == ec)
	{
		// connection closed cleanly by peer.
		isActive_ = false;
	}
	else if (ec)  // some other error.
	{
		//throw boost::system::system_error(ec);
		throw LogException(LogException::L_ERROR, ec.message(), __FILE__, __LINE__, __FUNCTION__);
	}

	return readLen;
}

}  // namespace swl
