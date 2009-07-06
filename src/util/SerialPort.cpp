#include "swl/Config.h"
#include "swl/util/SerialPort.h"
#include "swl/base/LogException.h"
#include "swl/base/String.h"
#include <boost/bind.hpp>
#include <iostream>


#if defined(_DEBUG)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


#if defined(min)
#undef min
#endif

namespace swl {

SerialPort::SerialPort(boost::asio::io_service &ioService)
: ioService_(ioService), port_(ioService), isActive_(false),
  receiveBuffer_(), sendBuffer_(), sentMsgLength_(0)
{}

SerialPort::~SerialPort()
{
	if (isActive_)
		disconnect();
}

#if defined(_UNICODE) || defined(UNICODE)
bool SerialPort::connect(const std::wstring &portName, const unsigned int baudRate)
#else
bool SerialPort::connect(const std::string &portName, const unsigned int baudRate)
#endif
{
#if defined(_UNICODE) || defined(UNICODE)
	port_.open(String::wcs2mbs(portName));
#else
	port_.open(portName);
#endif
	if (port_.is_open())
	{
		isActive_ = true;

		// options must be set after port was opened
		port_.set_option(boost::asio::serial_port::baud_rate(baudRate));
		port_.set_option(boost::asio::serial_port::flow_control(boost::asio::serial_port::flow_control::none));
		port_.set_option(boost::asio::serial_port::parity(boost::asio::serial_port::parity::none));
		port_.set_option(boost::asio::serial_port::stop_bits(boost::asio::serial_port::stop_bits::one));
		port_.set_option(boost::asio::serial_port::character_size(8));

		startReceiving();
		// TODO [check] >>
		if (!sendBuffer_.isEmpty()) startSending();

		return true;
	}
	else
	{ 
		std::cerr << "failed to open serial port" << std::endl;
		return false;
	} 
}

void SerialPort::disconnect()
{
	ioService_.post(boost::bind(&SerialPort::doCloseOperation, this, boost::system::error_code()));
}

void SerialPort::send(const unsigned char *msg, const size_t len)
{
	ioService_.post(boost::bind(&SerialPort::doSendOperation, this, msg, len));
}

size_t SerialPort::receive(unsigned char *msg, const size_t len)
{
	if (receiveBuffer_.isEmpty()) return 0;

	const size_t readLen = std::min(len, receiveBuffer_.getSize());
	receiveBuffer_.top(msg, readLen);
	receiveBuffer_.pop(readLen);
	return readLen;
}

void SerialPort::cancelIo()
{
	ioService_.post(boost::bind(&SerialPort::doCancelOperation, this, boost::system::error_code()));
}

void SerialPort::clearSendBuffer()
{
	sendBuffer_.clear();
}

void SerialPort::clearReceiveBuffer()
{
	receiveBuffer_.clear();
}

bool SerialPort::isSendBufferEmpty() const
{
	return sendBuffer_.isEmpty();
}

bool SerialPort::isReceiveBufferEmpty() const
{
	return receiveBuffer_.isEmpty();
}

size_t SerialPort::getSendBufferSize() const
{
	return sendBuffer_.getSize();
}

size_t SerialPort::getReceiveBufferSize() const
{
	return receiveBuffer_.getSize();
}

void SerialPort::doSendOperation(const unsigned char *msg, const size_t len)
{
	const bool write_in_progress = !sendBuffer_.isEmpty();
	sendBuffer_.push(msg, msg + len);
	if (!write_in_progress)
		startSending();
}

void SerialPort::startSending()
{
	sentMsgLength_ = std::min(sendBuffer_.getSize(), MAX_SEND_LENGTH_);
	sendBuffer_.top(sendMsg_, sentMsgLength_);
	boost::asio::async_write(
		port_,
		boost::asio::buffer(sendMsg_, sentMsgLength_),
		boost::bind(&SerialPort::completeSending, this, boost::asio::placeholders::error)
	);
}

void SerialPort::completeSending(const boost::system::error_code &ec)
{
	if (!ec)
	{
		sendBuffer_.pop(sentMsgLength_);
		sentMsgLength_ = 0;
		if (!sendBuffer_.isEmpty())
			startSending();
	}
	else
		doCloseOperation(ec);
}

void SerialPort::startReceiving()
{
	port_.async_read_some(
		boost::asio::buffer(receiveMsg_, MAX_RECEIVE_LENGTH_),
		boost::bind(&SerialPort::completeReceiving, this, boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred)
	); 
} 

void SerialPort::completeReceiving(const boost::system::error_code &ec, size_t bytesTransferred)
{
	if (!ec)
	{
		receiveBuffer_.push(receiveMsg_, bytesTransferred);
		startReceiving();
	}
	else
		doCloseOperation(ec);
}

void SerialPort::doCloseOperation(const boost::system::error_code &ec)
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

	port_.close();
	isActive_ = false;
}

void SerialPort::doCancelOperation(const boost::system::error_code &ec)
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

	port_.cancel();
}

}  // namespace swl
