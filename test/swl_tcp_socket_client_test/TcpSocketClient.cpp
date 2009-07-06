#include "swl/Config.h"
#include "TcpSocketClient.h"
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

size_t TcpSocketClient::send(const unsigned char *msg, const size_t len)
{
	boost::system::error_code ec;
	const size_t writtenLen = socket_.write_some(boost::asio::buffer(msg, len), ec);
	if (boost::asio::error::eof == ec)
	{
		// connection closed cleanly by peer.
		isActive_ = false;
	}
	else if (ec)  // some other error.
	{
		// FIXME [delete] >>
		const std::string &errMsg = ec.message();
		//throw boost::system::system_error(ec);
		throw LogException(LogException::L_ERROR, ec.message(), __FILE__, __LINE__, __FUNCTION__);
	}

	return writtenLen;
}

size_t TcpSocketClient::receive(unsigned char *msg, const size_t len)
{
	boost::system::error_code ec;
	const size_t readLen = socket_.read_some(boost::asio::buffer(msg, len), ec);
	if (boost::asio::error::eof == ec)
	{
		// connection closed cleanly by peer.
		isActive_ = false;
	}
	else if (ec)  // some other error.
	{
		// FIXME [delete] >>
		const std::string &errMsg = ec.message();
		//throw boost::system::system_error(ec);
		throw LogException(LogException::L_ERROR, ec.message(), __FILE__, __LINE__, __FUNCTION__);
	}

	return readLen;
}

//-----------------------------------------------------------------------------------
//

AsyncTcpSocketClient::AsyncTcpSocketClient(boost::asio::io_service &ioService)
: socket_(ioService), isActive_(false),
  isConnectDone_(false), isSendDone_(false), isReceiveDone_(false)
{
}

AsyncTcpSocketClient::~AsyncTcpSocketClient()
{
	if (isActive_)
		disconnect();
}

#if defined(_UNICODE) || defined(UNICODE)
bool AsyncTcpSocketClient::connect(const std::wstring &hostName, const std::wstring &serviceName)
#else
bool AsyncTcpSocketClient::connect(const std::string &hostName, const std::string &serviceName)
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
		//isConnectDone_ = false;
		//socket_.async_connect(*endpoint_iterator++, boost::bind(&AsyncTcpSocketClient::completeConnecting, this, boost::asio::placeholders::error));
	}
	if (ec) return false;

	isActive_ = true;
	return true;
}

void AsyncTcpSocketClient::disconnect()
{
	//socket_.close();
	//isActive_ = false;
	socket_.get_io_service().post(boost::bind(&AsyncTcpSocketClient::doCloseOperation, this, boost::system::error_code()));
}

void AsyncTcpSocketClient::send(const unsigned char *msg, const size_t len)
{
	sendMsg_.reserve(len);
	sendMsg_.assign(msg, msg + len);
	isSendDone_ = false;

	boost::asio::async_write(
		socket_,
		boost::asio::buffer(sendMsg_, len),
		boost::bind(&AsyncTcpSocketClient::completeSending, this, boost::asio::placeholders::error)
	);
}

void AsyncTcpSocketClient::receive(const size_t len)
{
	receiveMsg_.clear();
	receiveMsg_.reserve(len);
	isReceiveDone_ = false;

	boost::asio::async_read(
		socket_,
		boost::asio::buffer(receiveMsg_, len),
		boost::bind(&AsyncTcpSocketClient::completeReceiving, this, boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred)
	); 
}

void AsyncTcpSocketClient::getReceiveMessage(std::vector<unsigned char> &buf) const
{
	buf.reserve(receiveMsg_.size());
	buf.assign(receiveMsg_.begin(), receiveMsg_.end());
}

void AsyncTcpSocketClient::cancelIo()
{
	socket_.get_io_service().post(boost::bind(&AsyncTcpSocketClient::doCancelOperation, this, boost::system::error_code()));
}

void AsyncTcpSocketClient::completeConnecting(const boost::system::error_code &ec)
{
	if (!ec)
	{
		isConnectDone_ = true;
	}
	else
		doCloseOperation(ec);
}

void AsyncTcpSocketClient::completeSending(const boost::system::error_code &ec)
{
	if (!ec)
	{
		// FIXME [check] >> bytes transferred == len ???
		isSendDone_ = true;
		sendMsg_.clear();
	}
	else
		doCloseOperation(ec);
}

void AsyncTcpSocketClient::completeReceiving(const boost::system::error_code &ec, size_t bytesTransferred)
{
	if (!ec)
	{
		// FIXME [check] >> bytes transferred == len ???
		isReceiveDone_ = true;
	}
	else
		doCloseOperation(ec);
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
