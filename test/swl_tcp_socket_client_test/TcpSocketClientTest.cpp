#include "swl/Config.h"
#include "TcpSocketClient.h"
#include "swl/base/String.h"
#include <boost/asio.hpp>
#include <iostream>


#if defined(_DEBUG)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace {

const unsigned short serverPortNumer_FullDuplex = 5005;
const unsigned short serverPortNumer_HalfDuplex = 5006;

}  // unnamed namespace

#if defined(_UNICODE) || defined(UNICODE)
int wmain()
#else
int main()
#endif
{
	boost::asio::io_service ioService;
	swl::TcpSocketClient client(ioService);

	// full duplex TCP socket server
	std::cout << "==========================================================================" << std::endl;
	std::cout << "start full duplex TCP socket client" << std::endl;
	{
#if defined(_UNICODE) || defined(UNICODE)
		std::wostringstream stream;
#else
		std::ostringstream stream;
#endif
		stream << serverPortNumer_FullDuplex;
		const bool ret = client.connect(SWL_STR("localhost"), stream.str());
		if (ret != true)
			std::cerr << "***** connection error at " << __LINE__ << " in " << __FILE__ << std::endl;
		if (client.isConnected() != true)
			std::cerr << "***** connection check error at " << __LINE__ << " in " << __FILE__ << std::endl;
		if (ret)
		{
			{
				const unsigned char sendMsg[] = "abcdefghijklmnopqrstuvwxyz";
				unsigned char receiveMsg[256] = { '\0', };
				const size_t sendLen = client.send(sendMsg, std::strlen((char *)sendMsg) * sizeof(sendMsg[0]));
				if (std::strlen((char *)sendMsg) * sizeof(sendMsg[0]) != sendLen)
					std::cerr << "***** sending error at " << __LINE__ << " in " << __FILE__ << std::endl;
				const size_t receiveLen = client.receive(receiveMsg, sizeof(receiveMsg) * sizeof(receiveMsg[0]));
				if (std::strlen((char *)receiveMsg) * sizeof(sendMsg[0]) != receiveLen)
					std::cerr << "***** receiving error at " << __LINE__ << " in " << __FILE__ << std::endl;
				if (std::strcmp((char *)receiveMsg, (char *)sendMsg) != 0)
					std::cerr << "***** message error at " << __LINE__ << " in " << __FILE__ << std::endl;
			}

			{
				const unsigned char sendMsg[] = "9876543210";
				unsigned char receiveMsg[256] = { '\0', };
				const size_t sendLen = client.send(sendMsg, std::strlen((char *)sendMsg) * sizeof(sendMsg[0]));
				if (std::strlen((char *)sendMsg) * sizeof(sendMsg[0]) != sendLen)
					std::cerr << "***** sending error at " << __LINE__ << " in " << __FILE__ << std::endl;
				const size_t receiveLen = client.receive(receiveMsg, sizeof(receiveMsg) * sizeof(receiveMsg[0]));
				if (std::strlen((char *)receiveMsg) * sizeof(sendMsg[0]) != receiveLen)
					std::cerr << "***** receiving error at " << __LINE__ << " in " << __FILE__ << std::endl;
				if (std::strcmp((char *)receiveMsg, (char *)sendMsg) != 0)
					std::cerr << "***** message error at " << __LINE__ << " in " << __FILE__ << std::endl;
			}

			client.disconnect();
			if (client.isConnected() != false)
				std::cerr << "***** disconnection error at " << __LINE__ << " in " << __FILE__ << std::endl;
		}
	}
	std::cout << "finish half duplex TCP socket client" << std::endl;

	// half duplex TCP socket server
	std::cout << std::endl << "==========================================================================" << std::endl;
	std::cout << "start half duplex TCP socket client" << std::endl;
	{
#if defined(_UNICODE) || defined(UNICODE)
		std::wostringstream stream;
#else
		std::ostringstream stream;
#endif
		stream << serverPortNumer_HalfDuplex;
		const bool ret = client.connect(SWL_STR("localhost"), stream.str());
		if (ret != true)
			std::cerr << "***** connection error at " << __LINE__ << " in " << __FILE__ << std::endl;
		if (client.isConnected() != true)
			std::cerr << "***** connection check error at " << __LINE__ << " in " << __FILE__ << std::endl;
		if (ret)
		{
			{
				const unsigned char sendMsg[] = "abcdefghijklmnopqrstuvwxyz";
				unsigned char receiveMsg[256] = { '\0', };
				const size_t sendLen = client.send(sendMsg, std::strlen((char *)sendMsg) * sizeof(sendMsg[0]));
				if (std::strlen((char *)sendMsg) * sizeof(sendMsg[0]) != sendLen)
					std::cerr << "***** sending error at " << __LINE__ << " in " << __FILE__ << std::endl;
				const size_t receiveLen = client.receive(receiveMsg, sizeof(receiveMsg) * sizeof(receiveMsg[0]));
				if (std::strlen((char *)receiveMsg) * sizeof(sendMsg[0]) != receiveLen)
					std::cerr << "***** receiving error at " << __LINE__ << " in " << __FILE__ << std::endl;
				if (std::strcmp((char *)receiveMsg, (char *)sendMsg) != 0)
					std::cerr << "***** message error at " << __LINE__ << " in " << __FILE__ << std::endl;
			}

			{
				const unsigned char sendMsg[] = "9876543210";
				unsigned char receiveMsg[256] = { '\0', };
				const size_t sendLen = client.send(sendMsg, std::strlen((char *)sendMsg) * sizeof(sendMsg[0]));
				if (std::strlen((char *)sendMsg) * sizeof(sendMsg[0]) != sendLen)
					std::cerr << "***** sending error at " << __LINE__ << " in " << __FILE__ << std::endl;
				const size_t receiveLen = client.receive(receiveMsg, sizeof(receiveMsg) * sizeof(receiveMsg[0]));
				if (std::strlen((char *)receiveMsg) * sizeof(sendMsg[0]) != receiveLen)
					std::cerr << "***** receiving error at " << __LINE__ << " in " << __FILE__ << std::endl;
				if (std::strcmp((char *)receiveMsg, (char *)sendMsg) != 0)
					std::cerr << "***** message error at " << __LINE__ << " in " << __FILE__ << std::endl;
			}

			client.disconnect();
			if (client.isConnected() != false)
				std::cerr << "***** disconnection error at " << __LINE__ << " in " << __FILE__ << std::endl;
		}
	}
	std::cout << "finish half duplex TCP socket client" << std::endl;

	std::cout.flush();
	std::cin.get();
	return 0;
}
