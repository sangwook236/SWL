#include "swl/Config.h"
#include "swl/util/TcpSocketClient.h"
#include "swl/util/AsyncTcpSocketClient.h"
#include "swl/base/String.h"
#include <boost/asio.hpp>
#include <boost/thread.hpp>
#include <boost/smart_ptr.hpp>
#include <iostream>


#if defined(_DEBUG)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace {

const unsigned short serverPortNumber_withoutSession = 6000;
const unsigned short serverPortNumber_withSession = 7000;

}  // unnamed namespace

#if defined(_UNICODE) || defined(UNICODE)
int wmain()
#else
int main()
#endif
{
	{
		boost::asio::io_service ioService;
		swl::TcpSocketClient client(ioService);

		// synchronous TCP socket client that communicates with TCP socket server w/o session
		std::cout << "==========================================================================" << std::endl;
		std::cout << "start synchronous TCP socket client that communicates with TCP socket server w/o session" << std::endl;
		{
#if defined(_UNICODE) || defined(UNICODE)
			std::wostringstream stream;
#else
			std::ostringstream stream;
#endif
			stream << serverPortNumber_withoutSession;
			const bool ret = client.connect(SWL_STR("localhost"), stream.str());
			if (ret != true)
				std::cerr << "\t***** connection error at " << __LINE__ << " in " << __FILE__ << std::endl;
			if (ret)
			{
				if (client.isConnected() != true)
					std::cerr << "\t***** connection check error at " << __LINE__ << " in " << __FILE__ << std::endl;

				{
					const unsigned char sendMsg[] = "abcdefghijklmnopqrstuvwxyz";
					unsigned char receiveMsg[256] = { '\0', };
					const std::size_t sendingLen = std::strlen((char *)sendMsg) * sizeof(sendMsg[0]);
					const std::size_t maxReceivingLen = sizeof(receiveMsg) * sizeof(receiveMsg[0]);
					const std::size_t sentLen = client.send(sendMsg, sendingLen);
					if (sendingLen != sentLen)
						std::cerr << "\t***** sending error at " << __LINE__ << " in " << __FILE__ << std::endl;
					const std::size_t receivedLen = client.receive(receiveMsg, maxReceivingLen);
					if (std::strlen((char *)receiveMsg) * sizeof(receiveMsg[0]) != receivedLen)
						std::cerr << "\t***** receiving error at " << __LINE__ << " in " << __FILE__ << std::endl;
					if (std::strcmp((char *)receiveMsg, (char *)sendMsg) != 0)
						std::cerr << "\t***** message error at " << __LINE__ << " in " << __FILE__ << std::endl;
				}

				{
					const unsigned char sendMsg[] = "9876543210";
					unsigned char receiveMsg[256] = { '\0', };
					const std::size_t sendingLen = std::strlen((char *)sendMsg) * sizeof(sendMsg[0]);
					const std::size_t maxReceivingLen = sizeof(receiveMsg) * sizeof(receiveMsg[0]);
					const std::size_t sentLen = client.send(sendMsg, sendingLen);
					if (sendingLen != sentLen)
						std::cerr << "\t***** sending error at " << __LINE__ << " in " << __FILE__ << std::endl;
					const std::size_t receivedLen = client.receive(receiveMsg, maxReceivingLen);
					if (std::strlen((char *)receiveMsg) * sizeof(receiveMsg[0]) != receivedLen)
						std::cerr << "\t***** receiving error at " << __LINE__ << " in " << __FILE__ << std::endl;
					if (std::strcmp((char *)receiveMsg, (char *)sendMsg) != 0)
						std::cerr << "\t***** message error at " << __LINE__ << " in " << __FILE__ << std::endl;
				}

				client.disconnect();
				if (client.isConnected() != false)
					std::cerr << "\t***** disconnection error at " << __LINE__ << " in " << __FILE__ << std::endl;
			}
		}
		std::cout << "finish synchronous TCP socket client that communicates with TCP socket server w/o session" << std::endl;

		// synchronous TCP socket client that communicates with TCP socket server w/ session
		std::cout << std::endl << "==========================================================================" << std::endl;
		std::cout << "start synchronous TCP socket client that communicates with TCP socket server w/ session" << std::endl;
		{
#if defined(_UNICODE) || defined(UNICODE)
			std::wostringstream stream;
#else
			std::ostringstream stream;
#endif
			stream << serverPortNumber_withSession;
			const bool ret = client.connect(SWL_STR("localhost"), stream.str());
			if (ret != true)
				std::cerr << "\t***** connection error at " << __LINE__ << " in " << __FILE__ << std::endl;
			if (ret)
			{
				if (client.isConnected() != true)
					std::cerr << "\t***** connection check error at " << __LINE__ << " in " << __FILE__ << std::endl;

				{
					const unsigned char sendMsg[] = "abcdefghijklmnopqrstuvwxyz";
					unsigned char receiveMsg[256] = { '\0', };
					const std::size_t sendingLen = std::strlen((char *)sendMsg) * sizeof(sendMsg[0]);
					const std::size_t maxReceivingLen = sizeof(receiveMsg) * sizeof(receiveMsg[0]);
					const std::size_t sentLen = client.send(sendMsg, sendingLen);
					if (sendingLen != sentLen)
						std::cerr << "\t***** sending error at " << __LINE__ << " in " << __FILE__ << std::endl;
					const std::size_t receivedLen = client.receive(receiveMsg, maxReceivingLen);
					if (std::strlen((char *)receiveMsg) * sizeof(receiveMsg[0]) != receivedLen)
						std::cerr << "\t***** receiving error at " << __LINE__ << " in " << __FILE__ << std::endl;
					if (std::strcmp((char *)receiveMsg, (char *)sendMsg) != 0)
						std::cerr << "\t***** message error at " << __LINE__ << " in " << __FILE__ << std::endl;
				}

				{
					const unsigned char sendMsg[] = "9876543210";
					unsigned char receiveMsg[256] = { '\0', };
					const std::size_t sendingLen = std::strlen((char *)sendMsg) * sizeof(sendMsg[0]);
					const std::size_t maxReceivingLen = sizeof(receiveMsg) * sizeof(receiveMsg[0]);
					const std::size_t sentLen = client.send(sendMsg, sendingLen);
					if (sendingLen != sentLen)
						std::cerr << "\t***** sending error at " << __LINE__ << " in " << __FILE__ << std::endl;
					const std::size_t receivedLen = client.receive(receiveMsg, maxReceivingLen);
					if (std::strlen((char *)receiveMsg) * sizeof(receiveMsg[0]) != receivedLen)
						std::cerr << "\t***** receiving error at " << __LINE__ << " in " << __FILE__ << std::endl;
					if (std::strcmp((char *)receiveMsg, (char *)sendMsg) != 0)
						std::cerr << "\t***** message error at " << __LINE__ << " in " << __FILE__ << std::endl;
				}

				client.disconnect();
				if (client.isConnected() != false)
					std::cerr << "\t***** disconnection error at " << __LINE__ << " in " << __FILE__ << std::endl;
			}
		}
		std::cout << "finish synchronous TCP socket client that communicates with TCP socket server w/ session" << std::endl;
	}

	//----------------------------------------------------------------------------

	{
		const int MAX_ITER = 100;
		int idx = 0;

		// asynchronous TCP socket client that communicates with TCP socket server w/o session
		std::cout << std::endl << "==========================================================================" << std::endl;
		std::cout << "start asynchronous TCP socket client that communicates with TCP socket server w/o session" << std::endl;
		{
#if defined(_UNICODE) || defined(UNICODE)
			std::wostringstream stream;
#else
			std::ostringstream stream;
#endif
			stream << serverPortNumber_withoutSession;

			// caution: step 1 -> step 2 -> step 3
			// step 1
			boost::asio::io_service ioService;
			// step 2
			swl::AsyncTcpSocketClient client(ioService, SWL_STR("localhost"), stream.str());

			std::cout << "start thread for TCP socket client" << std::endl;
			// step 3
			//boost::thread clientWorkerThread(boost::bind(&boost::asio::io_service::run, &ioService)); 
			boost::scoped_ptr<boost::thread> clientWorkerThread(new boost::thread(boost::bind(&boost::asio::io_service::run, &ioService)));

			idx = 0;
			while (!client.isConnected() && idx < MAX_ITER)
			{
				boost::asio::deadline_timer timer(ioService);
				timer.expires_from_now(boost::posix_time::milliseconds(10));
				timer.wait();
				++idx;
			}
			if (MAX_ITER == idx)
				std::cerr << "\t***** connection error at " << __LINE__ << " in " << __FILE__ << std::endl;
			else
			{
				if (client.isConnected() != true)
					std::cerr << "\t***** connection check error at " << __LINE__ << " in " << __FILE__ << std::endl;
				if (client.isSendBufferEmpty() != true)
					std::cerr << "\t***** send buffer error at " << __LINE__ << " in " << __FILE__ << std::endl;
				if (client.isReceiveBufferEmpty() != true)
					std::cerr << "\t***** receive buffer error at " << __LINE__ << " in " << __FILE__ << std::endl;

				{
					const unsigned char sendMsg[] = "abcdefghijklmnopqrstuvwxyz";
					std::vector<unsigned char> receiveMsg;
					const std::size_t sendingLen = std::strlen((char *)sendMsg) * sizeof(sendMsg[0]);
					const std::size_t maxReceivingLen = sizeof(receiveMsg) * sizeof(receiveMsg[0]);
					client.send(sendMsg, sendingLen);
					idx = 0;
					while (client.getReceiveBufferSize() < sendingLen && idx < MAX_ITER)
					{
						boost::asio::deadline_timer timer(ioService);
						timer.expires_from_now(boost::posix_time::milliseconds(10));
						timer.wait();
						++idx;
					}
					if (MAX_ITER == idx)
						std::cerr << "\t***** receiving error at " << __LINE__ << " in " << __FILE__ << std::endl;

					std::vector<unsigned char> msg(sendingLen);
					client.receive(&msg[0], sendingLen);
					if (msg.empty() || std::strncmp((char *)&msg[0], (char *)sendMsg, sendingLen) != 0)
						std::cerr << "\t***** message error at " << __LINE__ << " in " << __FILE__ << std::endl;
				}

				if (client.isSendBufferEmpty() != true)
					std::cerr << "\t***** send buffer error at " << __LINE__ << " in " << __FILE__ << std::endl;
				if (client.isReceiveBufferEmpty() != true)
					std::cerr << "\t***** receive buffer error at " << __LINE__ << " in " << __FILE__ << std::endl;
				client.clearSendBuffer();
				client.clearReceiveBuffer();

				{
					const unsigned char sendMsg[] = "9876543210";
					std::vector<unsigned char> receiveMsg;
					const std::size_t sendingLen = std::strlen((char *)sendMsg) * sizeof(sendMsg[0]);
					const std::size_t maxReceivingLen = sizeof(receiveMsg) * sizeof(receiveMsg[0]);
					client.send(sendMsg, sendingLen);
					idx = 0;
					while (client.getReceiveBufferSize() < sendingLen && idx < MAX_ITER)
					{
						boost::asio::deadline_timer timer(ioService);
						timer.expires_from_now(boost::posix_time::milliseconds(10));
						timer.wait();
						++idx;
					}
					if (MAX_ITER == idx)
						std::cerr << "\t***** receiving error at " << __LINE__ << " in " << __FILE__ << std::endl;

					std::vector<unsigned char> msg(sendingLen);
					client.receive(&msg[0], sendingLen);
					if (msg.empty() || std::strncmp((char *)&msg[0], (char *)sendMsg, sendingLen) != 0)
						std::cerr << "\t***** message error at " << __LINE__ << " in " << __FILE__ << std::endl;
				}

				if (client.isSendBufferEmpty() != true)
					std::cerr << "\t***** send buffer error at " << __LINE__ << " in " << __FILE__ << std::endl;
				if (client.isReceiveBufferEmpty() != true)
					std::cerr << "\t***** receive buffer error at " << __LINE__ << " in " << __FILE__ << std::endl;

				client.disconnect();
				idx = 0;
				while (client.isConnected() && idx < MAX_ITER)
				{
					boost::asio::deadline_timer timer(ioService);
					timer.expires_from_now(boost::posix_time::milliseconds(10));
					timer.wait();
					++idx;
				}
				if (MAX_ITER == idx)
					std::cerr << "\t***** disconnection error at " << __LINE__ << " in " << __FILE__ << std::endl;
			}

			clientWorkerThread.reset();
			std::cout << "terminate thread for TCP socket client" << std::endl;
		}
		std::cout << "finish asynchronous TCP socket client that communicates with TCP socket server w/o session" << std::endl;

		// asynchronous TCP socket client that communicates with TCP socket server w/ session
		std::cout << std::endl << "==========================================================================" << std::endl;
		std::cout << "start asynchronous TCP socketclient that communicates with TCP socket server w/ session" << std::endl;
		{
#if defined(_UNICODE) || defined(UNICODE)
			std::wostringstream stream;
#else
			std::ostringstream stream;
#endif
			stream << serverPortNumber_withSession;

			// caution: step 1 -> step 2 -> step 3
			// step 1
			boost::asio::io_service ioService;
			// step 2
			swl::AsyncTcpSocketClient client(ioService, SWL_STR("localhost"), stream.str());

			std::cout << "start thread for TCP socket client" << std::endl;
			// step 3
			//boost::thread clientWorkerThread(boost::bind(&boost::asio::io_service::run, &ioService)); 
			boost::scoped_ptr<boost::thread> clientWorkerThread(new boost::thread(boost::bind(&boost::asio::io_service::run, &ioService)));

			idx = 0;
			while (!client.isConnected() && idx < MAX_ITER)
			{
				boost::asio::deadline_timer timer(ioService);
				timer.expires_from_now(boost::posix_time::milliseconds(10));
				timer.wait();
				++idx;
			}
			if (MAX_ITER == idx)
				std::cerr << "\t***** connection error at " << __LINE__ << " in " << __FILE__ << std::endl;
			else
			{
				if (client.isConnected() != true)
					std::cerr << "\t***** connection check error at " << __LINE__ << " in " << __FILE__ << std::endl;
				if (client.isSendBufferEmpty() != true)
					std::cerr << "\t***** send buffer error at " << __LINE__ << " in " << __FILE__ << std::endl;
				if (client.isReceiveBufferEmpty() != true)
					std::cerr << "\t***** receive buffer error at " << __LINE__ << " in " << __FILE__ << std::endl;

				{
					const unsigned char sendMsg[] = "abcdefghijklmnopqrstuvwxyz";
					std::vector<unsigned char> receiveMsg;
					const std::size_t sendingLen = std::strlen((char *)sendMsg) * sizeof(sendMsg[0]);
					const std::size_t maxReceivingLen = sizeof(receiveMsg) * sizeof(receiveMsg[0]);
					client.send(sendMsg, sendingLen);
					idx = 0;
					while (client.getReceiveBufferSize() < sendingLen && idx < MAX_ITER)
					{
						boost::asio::deadline_timer timer(ioService);
						timer.expires_from_now(boost::posix_time::milliseconds(10));
						timer.wait();
						++idx;
					}
					if (MAX_ITER == idx)
						std::cerr << "\t***** receiving error at " << __LINE__ << " in " << __FILE__ << std::endl;

					std::vector<unsigned char> msg(sendingLen);
					client.receive(&msg[0], sendingLen);
					if (msg.empty() || std::strncmp((char *)&msg[0], (char *)sendMsg, sendingLen) != 0)
						std::cerr << "\t***** message error at " << __LINE__ << " in " << __FILE__ << std::endl;
				}

				if (client.isSendBufferEmpty() != true)
					std::cerr << "\t***** send buffer error at " << __LINE__ << " in " << __FILE__ << std::endl;
				if (client.isReceiveBufferEmpty() != true)
					std::cerr << "\t***** receive buffer error at " << __LINE__ << " in " << __FILE__ << std::endl;
				client.clearSendBuffer();
				client.clearReceiveBuffer();

				{
					const unsigned char sendMsg[] = "9876543210";
					std::vector<unsigned char> receiveMsg;
					const std::size_t sendingLen = std::strlen((char *)sendMsg) * sizeof(sendMsg[0]);
					const std::size_t maxReceivingLen = sizeof(receiveMsg) * sizeof(receiveMsg[0]);
					client.send(sendMsg, sendingLen);
					idx = 0;
					while (client.getReceiveBufferSize() < sendingLen && idx < MAX_ITER)
					{
						boost::asio::deadline_timer timer(ioService);
						timer.expires_from_now(boost::posix_time::milliseconds(10));
						timer.wait();
						++idx;
					}
					if (MAX_ITER == idx)
						std::cerr << "\t***** receiving error at " << __LINE__ << " in " << __FILE__ << std::endl;

					std::vector<unsigned char> msg(sendingLen);
					client.receive(&msg[0], sendingLen);
					if (msg.empty() || std::strncmp((char *)&msg[0], (char *)sendMsg, sendingLen) != 0)
						std::cerr << "\t***** message error at " << __LINE__ << " in " << __FILE__ << std::endl;
				}

				if (client.isSendBufferEmpty() != true)
					std::cerr << "\t***** send buffer error at " << __LINE__ << " in " << __FILE__ << std::endl;
				if (client.isReceiveBufferEmpty() != true)
					std::cerr << "\t***** receive buffer error at " << __LINE__ << " in " << __FILE__ << std::endl;

				client.disconnect();
				idx = 0;
				while (client.isConnected() && idx < MAX_ITER)
				{
					boost::asio::deadline_timer timer(ioService);
					timer.expires_from_now(boost::posix_time::milliseconds(10));
					timer.wait();
					++idx;
				}
				if (MAX_ITER == idx)
					std::cerr << "\t***** disconnection error at " << __LINE__ << " in " << __FILE__ << std::endl;
			}

			clientWorkerThread.reset();
			std::cout << "terminate thread for TCP socket client" << std::endl;
		}
		std::cout << "finish asynchronous TCP socket client that communicates with TCP socket server w/ session" << std::endl;
	}

	std::cout.flush();
	std::cin.get();
	return 0;
}
