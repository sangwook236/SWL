#include "swl/Config.h"
#include "swl/util/TcpSocketClient.h"
#include "swl/util/AsyncTcpSocketClient.h"
#include "swl/base/String.h"
#include <boost/asio.hpp>
#include <boost/thread.hpp>
#include <boost/smart_ptr.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cstdlib>


#if defined(_DEBUG)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace {

const unsigned short serverPortNumber_withoutSession = 6000;
const unsigned short serverPortNumber_withSession = 7000;

const unsigned char CHARSET[] = {
	'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
	'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
	'0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
	'~', '`', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '_', '=', '+', '\\', '|',
	',', '<', '.', '>', '/', '?', ';', ':', '\'', '\"', '[', '{', ']', '}',
};

const std::size_t CHARSET_LEN = sizeof(CHARSET);

boost::mutex io_mutex;  // The iostreams are not guaranteed to be thread-safe
const std::size_t NUM_THREADS = 1000;

struct tcp_socket_client_worker_thread_functor
{
	tcp_socket_client_worker_thread_functor(boost::asio::io_service &ioService, const swl::string_t &hostName, const swl::string_t &serviceName)
	: ioService_(ioService), hostName_(hostName), serviceName_(serviceName)
	{}

	void operator()()
	{
		std::srand((unsigned int)std::time(NULL));

		try
		{
			swl::TcpSocketClient client(ioService_);
			if (client.connect(hostName_, serviceName_))
			{
				std::list<unsigned char> sendBuf;
				std::list<unsigned char> receiveBuf;

				while (true)
				{
					const std::size_t sendingLen = std::rand();
					if (sendingLen)
					{
						std::vector<unsigned char> sendMsg;
						sendMsg.reserve(sendingLen);
						for (std::size_t i = 0; i < sendingLen; ++i)
							sendMsg.push_back(CHARSET[std::rand() % CHARSET_LEN]);

						std::vector<unsigned char> receiveMsg(sendingLen, '\0');

						const std::size_t sentLen = client.send(&sendMsg[0], sendingLen);
						std::cout << '\t' << boost::this_thread::get_id() << ">>>>> send: "; std::cout.write((char *)&sendMsg[0], (std::streamsize)sendingLen); std::cout << std::endl;
						//if (sendingLen != sentLen)
						//	throw std::runtime_error("send error: length mismatch");
						const std::size_t receivedLen = client.receive(&receiveMsg[0], sendingLen);
						std::cout << '\t' << boost::this_thread::get_id() << "<<<<< receive: "; std::cout.write((char *)&receiveMsg[0], (std::streamsize)receivedLen); std::cout << std::endl;
						//if (sendingLen != receivedLen)
						//	throw std::runtime_error("receive error: length mismatch");
						//if (std::strncmp((char *)&receiveMsg[0], (char *)&sendMsg[0], sendingLen) != 0)
						//	throw std::runtime_error("message error: message mismatch");

						std::copy(sendMsg.begin(), sendMsg.end(), std::back_inserter(sendBuf));
						std::vector<unsigned char>::iterator itEnd = receiveMsg.begin();
						std::advance(itEnd, receivedLen);
						std::copy(receiveMsg.begin(), itEnd, std::back_inserter(receiveBuf));
					}

					match(sendBuf, receiveBuf);

					const std::size_t waitingTime = std::rand() % 5 + 1;  // [sec]
					boost::xtime xt;
					boost::xtime_get(&xt, boost::TIME_UTC);
					xt.sec += waitingTime;
					//xt.nsec += waitingTime;
					boost::thread::sleep(xt);

					boost::thread::yield();
				}
			}
			else
			{
				throw std::runtime_error("connection error: connection fail");
			}
		}
		catch (const std::runtime_error &e)
		{
			report(e.what());
		}
		catch (const std::exception &e)
		{
			report(e.what());
		}
		catch (...)
		{
			report("unknown exception occurred");
		}
	}

private:
	void match(std::list<unsigned char> &sendBuf, std::list<unsigned char> &receiveBuf) const
	{
		std::list<unsigned char>::iterator itSend = sendBuf.begin();
		std::list<unsigned char>::iterator itReceive = receiveBuf.begin();
		for (; itSend != sendBuf.end() && itReceive != receiveBuf.end(); )
		{
			if (*itSend == *itReceive)
			{
				//const std::size_t s = sendBuf.size();
				//const std::size_t r = receiveBuf.size();
				itSend = sendBuf.erase(itSend);
				itReceive = receiveBuf.erase(itReceive);
			}
			else
				throw std::runtime_error("message error: message mismatch");
		}
	}

	void report(const std::string &msg) const
	{
		boost::mutex::scoped_lock lock(io_mutex);

		std::cout << msg << std::endl;

		std::ofstream stream("test_result\\tcp_socket_client_test_result.txt", std::ios::out | std::ios::app);
		if (stream.is_open())
		{
			stream << "TCP socket client worker thread, " << boost::this_thread::get_id() << " is terminated" << std::endl;
			stream << '\t' << __TIMESTAMP__ << ", " << msg << std::endl;
		}
	}

private:
	boost::asio::io_service &ioService_;
	const swl::string_t hostName_;
	const swl::string_t serviceName_;
};

}  // unnamed namespace

void testSyncTcpSocketClient();
void testAsyncTcpSocketClient();
void testTcpSocketServer_withoutSession();
void testTcpSocketServer_withSession();

#if defined(_UNICODE) || defined(UNICODE)
int wmain()
#else
int main()
#endif
{
	//
	//testSyncTcpSocketClient();
	//testAsyncTcpSocketClient();

	//
	testTcpSocketServer_withoutSession();
	//testTcpSocketServer_withSession();

	std::cout.flush();
	std::cin.get();
	return 0;
}

void testSyncTcpSocketClient()
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

void testAsyncTcpSocketClient()
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

void testTcpSocketServer_withoutSession()
{
	boost::asio::io_service ioService;

#if defined(_UNICODE) || defined(UNICODE)
	std::wostringstream stream;
#else
	std::ostringstream stream;
#endif
	stream << serverPortNumber_withoutSession;

    boost::thread_group thrds;
    for (std::size_t i = 0; i < NUM_THREADS; ++i)
        thrds.create_thread(tcp_socket_client_worker_thread_functor(ioService, SWL_STR("localhost"), stream.str()));

    thrds.join_all();
}

void testTcpSocketServer_withSession()
{
	boost::asio::io_service ioService;

#if defined(_UNICODE) || defined(UNICODE)
	std::wostringstream stream;
#else
	std::ostringstream stream;
#endif
	stream << serverPortNumber_withSession;

	std::cout << "create thread group of size " << NUM_THREADS << std::endl;
    boost::thread_group thrds;
    for (std::size_t i = 0; i < NUM_THREADS; ++i)
        thrds.create_thread(tcp_socket_client_worker_thread_functor(ioService, SWL_STR("localhost"), stream.str()));

	std::cout << "start thread group of size " << thrds.size() << std::endl;
    thrds.join_all();
	std::cout << "finish thread group" << std::endl;
}
