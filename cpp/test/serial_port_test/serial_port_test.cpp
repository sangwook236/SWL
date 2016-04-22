//#include "stdafx.h"
#include "swl/Config.h"
#include "swl/util/SerialPort.h"
#if defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64)
#include "swl/winutil/WinSerialPort.h"
#endif
#include <boost/thread.hpp>
#include <boost/smart_ptr.hpp>
#include <iostream>
#include <ctime>
#include <stdexcept>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace {
namespace local {

struct BoostSerialPortThreadFunctor
{
	BoostSerialPortThreadFunctor(boost::asio::io_service &ioService)
	: ioService_(ioService)
	{}
	~BoostSerialPortThreadFunctor()
	{}

public:
	void operator()()
	{
		std::cout << "boost serial port worker thread is started" << std::endl;
		ioService_.run();
		std::cout << "boost serial port worker thread is terminated" << std::endl;
	}

private:
	boost::asio::io_service &ioService_;
};

#if defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64)
struct WinSerialPortThreadFunctor
{
	WinSerialPortThreadFunctor(swl::WinSerialPort &serialPort, swl::GuardedByteBuffer &recvBuffer)
	: serialPort_(serialPort), recvBuffer_(recvBuffer)
	{}
	~WinSerialPortThreadFunctor()
	{}

public:
	void operator()()
	{
		std::cout << "win serial port worker thread is started" << std::endl;

		const unsigned long timeoutInterval_msec = 10;
		const size_t bufferLen = 0;
		while (true)
		{
			serialPort_.receive(recvBuffer_, timeoutInterval_msec, bufferLen);

			boost::this_thread::yield();
		}

		std::cout << "win serial port worker thread is terminated" << std::endl;
	}

private:
	swl::WinSerialPort &serialPort_;
	swl::GuardedByteBuffer &recvBuffer_;
};
#endif

}  // namespace local
}  // unnamed namespace

void test_boost_serial_port()
{
	boost::asio::io_service ioService;
	swl::SerialPort serialPort(ioService);

#if defined(_UNICODE) || defined(UNICODE)
	const std::wstring portName = L"COM1";
#else
	const std::string portName = "COM1";
#endif
	const unsigned int baudRate = 9600;

	if (serialPort.connect(portName, baudRate))
	{
#if defined(_UNICODE) || defined(UNICODE)
		std::wcout << L"serial port connected ==> port : " << portName << L", baud-rate : " << baudRate << std::endl;
#else
		std::cout << "serial port connected ==> port : " << portName << ", baud-rate : " << baudRate << std::endl;
#endif

		// create boost serial port worker thread
		boost::scoped_ptr<boost::thread> workerThread;
		workerThread.reset(new boost::thread(local::BoostSerialPortThreadFunctor(ioService)));

		std::srand((unsigned int)time(NULL));

		const size_t msgLen = 4095;
		unsigned char msg[msgLen + 1] = { '\0', };
		while (true)
		{
			if (!serialPort.isReceiveBufferEmpty())
			{
				memset(msg, 0, msgLen + 1);

				serialPort.receive(msg, msgLen);
				std::cout << msg << std::endl;
			}

			if (0 == std::rand() % 10)
			{
				const std::string sendMsg("a test message is just sent");
				serialPort.send((unsigned char *)sendMsg.c_str(), sendMsg.length());
			}

			boost::this_thread::yield();
		}

#if defined(_UNICODE) || defined(UNICODE)
		std::wcout << L"serial port disconnected ==> port : " << portName << L", baud-rate : " << baudRate << std::endl;
#else
		std::cout << "serial port disconnected ==> port : " << portName << ", baud-rate : " << baudRate << std::endl;
#endif

		workerThread.reset();
		serialPort.disconnect();
	}
}

void test_windows_serial_port()
{
#if defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64)
	swl::WinSerialPort serialPort;

#if defined(_UNICODE) || defined(UNICODE)
	const std::wstring portName = L"COM1";
#else
	const std::string portName = "COM1";
#endif
	const unsigned int baudRate = 9600;

	const size_t inQueueSize = 8192, outQueueSize = 8192;
	if (serialPort.connect(portName.c_str(), baudRate, inQueueSize, outQueueSize))
	{
#if defined(_UNICODE) || defined(UNICODE)
		std::wcout << L"serial port connected ==> port : " << portName << L", baud-rate : " << baudRate << std::endl;
#else
		std::cout << "serial port connected ==> port : " << portName << ", baud-rate : " << baudRate << std::endl;
#endif

		swl::GuardedByteBuffer recvBuffer;

		// create boost serial port worker thread
		boost::scoped_ptr<boost::thread> workerThread;
		workerThread.reset(new boost::thread(local::WinSerialPortThreadFunctor(serialPort, recvBuffer)));

		std::srand((unsigned int)time(NULL));

		const size_t msgLen = 4095;
		unsigned char msg[msgLen + 1] = { '\0', };
		while (true)
		{
			if (!recvBuffer.isEmpty())
			{
				memset(msg, 0, msgLen + 1);

				const size_t len = recvBuffer.getSize();
				const size_t len2 = len > msgLen ? msgLen : len;
				recvBuffer.top(msg, len2);
				recvBuffer.pop(len2);
				std::cout << msg << std::endl;
			}

			if (0 == std::rand() % 10)
			{
				const std::string sendMsg("a test message is just sent");
				serialPort.send((unsigned char *)sendMsg.c_str(), sendMsg.length());
			}

			Sleep(0);
		}

#if defined(_UNICODE) || defined(UNICODE)
		std::wcout << L"serial port disconnected ==> port : " << portName << L", baud-rate : " << baudRate << std::endl;
#else
		std::cout << "serial port disconnected ==> port : " << portName << ", baud-rate : " << baudRate << std::endl;
#endif

		workerThread.reset();
		serialPort.disconnect();
	}
#else
    throw std::runtime_error("this function is supported in WIN32 only");
#endif
}
