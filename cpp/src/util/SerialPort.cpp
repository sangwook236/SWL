#include "swl/Config.h"
#include "swl/util/SerialPort.h"
#include "swl/base/LogException.h"
#include "swl/base/String.h"
#include <boost/bind.hpp>
#include <iostream>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


#if defined(min)
#undef min
#endif


namespace swl {

SerialPort::SerialPort(boost::asio::io_service &ioService)
: port_(ioService), isActive_(false),
  sendBuffer_(), receiveBuffer_(), sendMsg_(), receiveMsg_(), sentMsgLength_(0)
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

		doStartReceiving();
		// TODO [check] >>
		if (!sendBuffer_.isEmpty()) doStartSending();

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
	port_.get_io_service().post(boost::bind(&SerialPort::doCloseOperation, this, boost::system::error_code()));
}

void SerialPort::send(const unsigned char *msg, const std::size_t len)
{
	port_.get_io_service().post(boost::bind(&SerialPort::doSendOperation, this, msg, len));
}

std::size_t SerialPort::receive(unsigned char *msg, const std::size_t len)
{
	if (receiveBuffer_.isEmpty()) return 0;

	const std::size_t readLen = std::min(len, receiveBuffer_.getSize());
	receiveBuffer_.top(msg, readLen);
	receiveBuffer_.pop(readLen);
	return readLen;
}

void SerialPort::cancelIo()
{
	port_.get_io_service().post(boost::bind(&SerialPort::doCancelOperation, this, boost::system::error_code()));
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

std::size_t SerialPort::getSendBufferSize() const
{
	return sendBuffer_.getSize();
}

std::size_t SerialPort::getReceiveBufferSize() const
{
	return receiveBuffer_.getSize();
}

void SerialPort::doStartSending()
{
#if defined(__GNUC__)
	sentMsgLength_ = std::min(sendBuffer_.getSize(), (std::size_t)MAX_SEND_LENGTH_);
#else
	sentMsgLength_ = std::min(sendBuffer_.getSize(), MAX_SEND_LENGTH_);
#endif
	sendBuffer_.top(sendMsg_.c_array(), sentMsgLength_);
	boost::asio::async_write(
		port_,
		boost::asio::buffer(sendMsg_, sentMsgLength_),
		boost::bind(&SerialPort::doCompleteSending, this, boost::asio::placeholders::error)
	);
}

void SerialPort::doCompleteSending(const boost::system::error_code &ec)
{
	if (!ec)
	{
		sendBuffer_.pop(sentMsgLength_);
		sentMsgLength_ = 0;
		if (!sendBuffer_.isEmpty())
			doStartSending();
	}
	else
		doCloseOperation(ec);
}

void SerialPort::doStartReceiving()
{
	port_.async_read_some(
		boost::asio::buffer(receiveMsg_),
		boost::bind(&SerialPort::doCompleteReceiving, this, boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred)
	);
}

void SerialPort::doCompleteReceiving(const boost::system::error_code &ec, std::size_t bytesTransferred)
{
	if (!ec)
	{
		receiveBuffer_.push(receiveMsg_.data(), bytesTransferred);
		doStartReceiving();
	}
	else
		doCloseOperation(ec);
}

void SerialPort::doSendOperation(const unsigned char *msg, const std::size_t len)
{
	const bool write_in_progress = !sendBuffer_.isEmpty();
	sendBuffer_.push(msg, msg + len);
	if (!write_in_progress)
		doStartSending();
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
