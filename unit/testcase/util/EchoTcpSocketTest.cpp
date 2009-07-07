#include "swl/Config.h"
#include "EchoTcpSocketConnection.h"
#include "EchoTcpSocketSession.h"
#include "../../UnitTestConfig.h"
#include "swl/util/TcpSocketClient.h"
#include "swl/util/AsyncTcpSocketClient.h"
#include "swl/util/TcpSocketServer.h"
#include "swl/util/TcpSocketConnection.h"
#include "swl/util/TcpSocketConnectionUsingSession.h"
#include "swl/base/String.h"
#include <boost/thread.hpp>
#include <boost/smart_ptr.hpp>
#include <sstream>

#if defined(_DEBUG)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {
namespace unit_test {

namespace {

const unsigned short serverPortNumber_withoutSession = 6000;
const unsigned short serverPortNumber_withSession = 7000;

struct echo_tcp_socket_server_worker_thread_functor
{
	void operator()()
	{
		boost::asio::io_service ioService;

		swl::TcpSocketServer<swl::EchoTcpSocketConnection> server(ioService, serverPortNumber_withoutSession);
		swl::TcpSocketServer<swl::TcpSocketConnectionUsingSession<swl::EchoTcpSocketSession> > sessionServer(ioService, serverPortNumber_withSession);

#if defined(__SWL_UNIT_TEST__USE_BOOST_UNIT)
		BOOST_TEST_MESSAGE("start TCP socket servers: w/o & w/ session");
#endif
		ioService.run();
#if defined(__SWL_UNIT_TEST__USE_BOOST_UNIT)
		BOOST_TEST_MESSAGE("finish TCP socket servers: w/o & w/ session");
#endif
	}
};

}  // unnamed namespace

//-----------------------------------------------------------------------------
//

#if defined(__SWL_UNIT_TEST__USE_BOOST_UNIT)

namespace {

struct EchoTcpSocketTest
{
private:
	struct Fixture
	{
		Fixture()  // set up
		{
			BOOST_TEST_MESSAGE("start thread for TCP socket servers");
			thrd_.reset(new boost::thread(echo_tcp_socket_server_worker_thread_functor()));
		}

		~Fixture()  // tear down
		{
			thrd_.reset();
			BOOST_TEST_MESSAGE("terminate thread for TCP socket servers");
		}

		boost::scoped_ptr<boost::thread> thrd_;
	};

public:
	void testSyncClient()
	{
		Fixture fixture;

		boost::asio::io_service ioService;
		swl::TcpSocketClient client(ioService);

		// synchronous TCP socket client that communicates with TCP socket server w/o session
		BOOST_TEST_MESSAGE("start synchronous TCP socket client that communicates with TCP socket server w/o session");
		{
#if defined(_UNICODE) || defined(UNICODE)
			std::wostringstream stream;
#else
			std::ostringstream stream;
#endif
			stream << serverPortNumber_withoutSession;
			const bool ret = client.connect(SWL_STR("localhost"), stream.str());
			BOOST_CHECK(ret);
			if (ret)
			{
				BOOST_CHECK_EQUAL(client.isConnected(), true);

				{
					const unsigned char sendMsg[] = "abcdefghijklmnopqrstuvwxyz";
					unsigned char receiveMsg[256] = { '\0', };
					const std::size_t sendingLen = std::strlen((char *)sendMsg) * sizeof(sendMsg[0]);
					const std::size_t maxReceivingLen = sizeof(receiveMsg) * sizeof(receiveMsg[0]);
					const size_t sentLen = client.send(sendMsg, sendingLen);
					BOOST_CHECK_EQUAL(sendingLen, sentLen);
					const size_t receivedLen = client.receive(receiveMsg, maxReceivingLen);
					BOOST_CHECK_EQUAL(std::strlen((char *)receiveMsg) * sizeof(receiveMsg[0]), receivedLen);
					BOOST_CHECK(std::strcmp((char *)receiveMsg, (char *)sendMsg) == 0);
				}

				{
					const unsigned char sendMsg[] = "9876543210";
					unsigned char receiveMsg[256] = { '\0', };
					const std::size_t sendingLen = std::strlen((char *)sendMsg) * sizeof(sendMsg[0]);
					const std::size_t maxReceivingLen = sizeof(receiveMsg) * sizeof(receiveMsg[0]);
					const size_t sentLen = client.send(sendMsg, sendingLen);
					BOOST_CHECK_EQUAL(sendingLen, sentLen);
					const size_t receivedLen = client.receive(receiveMsg, maxReceivingLen);
					BOOST_CHECK_EQUAL(std::strlen((char *)receiveMsg) * sizeof(receiveMsg[0]), receivedLen);
					BOOST_CHECK(std::strcmp((char *)receiveMsg, (char *)sendMsg) == 0);
				}

				client.disconnect();
				BOOST_CHECK_EQUAL(client.isConnected(), false);
			}
		}
		BOOST_TEST_MESSAGE("finish synchronous TCP socket client that communicates with TCP socket server w/o session");

		// synchronous TCP socket client that communicates with TCP socket server using session
		BOOST_TEST_MESSAGE("start synchronous TCP socket client that communicates with TCP socket server w/ session");
		{
#if defined(_UNICODE) || defined(UNICODE)
			std::wostringstream stream;
#else
			std::ostringstream stream;
#endif
			stream << serverPortNumber_withSession;
			const bool ret = client.connect(SWL_STR("localhost"), stream.str());
			BOOST_CHECK(ret);
			if (ret)
			{
				BOOST_CHECK_EQUAL(client.isConnected(), true);

				{
					const unsigned char sendMsg[] = "abcdefghijklmnopqrstuvwxyz";
					unsigned char receiveMsg[256] = { '\0', };
					const std::size_t sendingLen = std::strlen((char *)sendMsg) * sizeof(sendMsg[0]);
					const std::size_t maxReceivingLen = sizeof(receiveMsg) * sizeof(receiveMsg[0]);
					const size_t sentLen = client.send(sendMsg, sendingLen);
					BOOST_CHECK_EQUAL(sendingLen, sentLen);
					const size_t receivedLen = client.receive(receiveMsg, maxReceivingLen);
					BOOST_CHECK_EQUAL(std::strlen((char *)receiveMsg) * sizeof(receiveMsg[0]), receivedLen);
					BOOST_CHECK(std::strcmp((char *)receiveMsg, (char *)sendMsg) == 0);
				}

				{
					const unsigned char sendMsg[] = "9876543210";
					unsigned char receiveMsg[256] = { '\0', };
					const std::size_t sendingLen = std::strlen((char *)sendMsg) * sizeof(sendMsg[0]);
					const std::size_t maxReceivingLen = sizeof(receiveMsg) * sizeof(receiveMsg[0]);
					const size_t sentLen = client.send(sendMsg, sendingLen);
					BOOST_CHECK_EQUAL(sendingLen, sentLen);
					const size_t receivedLen = client.receive(receiveMsg, maxReceivingLen);
					BOOST_CHECK_EQUAL(std::strlen((char *)receiveMsg) * sizeof(receiveMsg[0]), receivedLen);
					BOOST_CHECK(std::strcmp((char *)receiveMsg, (char *)sendMsg) == 0);
				}

				client.disconnect();
				BOOST_CHECK_EQUAL(client.isConnected(), false);
			}
		}
		BOOST_TEST_MESSAGE("finish synchronous TCP socket client that communicates with TCP socket server w/ session");
	}

	void testAsyncClient()
	{
		Fixture fixture;

		const int MAX_ITER = 100;
		int idx = 0;

		// asynchronous TCP socket client that communicates with TCP socket server w/o session
		BOOST_TEST_MESSAGE("start asynchronous TCP socket client that communicates with TCP socket server w/o session");
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

			BOOST_TEST_MESSAGE("start thread for TCP socket client");
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
			BOOST_CHECK(MAX_ITER != idx);

			{
				BOOST_CHECK_EQUAL(client.isConnected(), true);
				BOOST_CHECK_EQUAL(client.isSendBufferEmpty(), true);
				BOOST_CHECK_EQUAL(client.isReceiveBufferEmpty(), true);

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
					BOOST_CHECK(MAX_ITER != idx);

					std::vector<unsigned char> msg(sendingLen);
					client.receive(&msg[0], sendingLen);
					BOOST_CHECK(!msg.empty());
					BOOST_CHECK(std::strncmp((char *)&msg[0], (char *)sendMsg, sendingLen) == 0);
				}

				BOOST_CHECK_EQUAL(client.isSendBufferEmpty(), true);
				BOOST_CHECK_EQUAL(client.isReceiveBufferEmpty(), true);
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
					BOOST_CHECK(MAX_ITER != idx);

					std::vector<unsigned char> msg(sendingLen);
					client.receive(&msg[0], sendingLen);
					BOOST_CHECK(!msg.empty());
					BOOST_CHECK(std::strncmp((char *)&msg[0], (char *)sendMsg, sendingLen) == 0);
				}

				BOOST_CHECK_EQUAL(client.isSendBufferEmpty(), true);
				BOOST_CHECK_EQUAL(client.isReceiveBufferEmpty(), true);

				client.disconnect();
				idx = 0;
				while (client.isConnected() && idx < MAX_ITER)
				{
					boost::asio::deadline_timer timer(ioService);
					timer.expires_from_now(boost::posix_time::milliseconds(10));
					timer.wait();
					++idx;
				}
				BOOST_CHECK(MAX_ITER != idx);
				BOOST_CHECK_EQUAL(client.isConnected(), false);
			}

			clientWorkerThread.reset();
			BOOST_TEST_MESSAGE("terminate thread for TCP socket client");
		}
		BOOST_TEST_MESSAGE("finish asynchronous TCP socket client that communicates with TCP socket server w/o session");

		// asynchronous TCP socket client that communicates with TCP socket server using session
		BOOST_TEST_MESSAGE("start asynchronous TCP socket client that communicates with TCP socket server w/ session");
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

			BOOST_TEST_MESSAGE("start thread for TCP socket client");
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
			BOOST_CHECK(MAX_ITER != idx);

			{
				BOOST_CHECK_EQUAL(client.isConnected(), true);
				BOOST_CHECK_EQUAL(client.isSendBufferEmpty(), true);
				BOOST_CHECK_EQUAL(client.isReceiveBufferEmpty(), true);

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
					BOOST_CHECK(MAX_ITER != idx);

					std::vector<unsigned char> msg(sendingLen);
					client.receive(&msg[0], sendingLen);
					BOOST_CHECK(!msg.empty());
					BOOST_CHECK(std::strncmp((char *)&msg[0], (char *)sendMsg, sendingLen) == 0);
				}

				BOOST_CHECK_EQUAL(client.isSendBufferEmpty(), true);
				BOOST_CHECK_EQUAL(client.isReceiveBufferEmpty(), true);
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
					BOOST_CHECK(MAX_ITER != idx);

					std::vector<unsigned char> msg(sendingLen);
					client.receive(&msg[0], sendingLen);
					BOOST_CHECK(!msg.empty());
					BOOST_CHECK(std::strncmp((char *)&msg[0], (char *)sendMsg, sendingLen) == 0);
				}

				BOOST_CHECK_EQUAL(client.isSendBufferEmpty(), true);
				BOOST_CHECK_EQUAL(client.isReceiveBufferEmpty(), true);

				client.disconnect();
				idx = 0;
				while (client.isConnected() && idx < MAX_ITER)
				{
					boost::asio::deadline_timer timer(ioService);
					timer.expires_from_now(boost::posix_time::milliseconds(10));
					timer.wait();
					++idx;
				}
				BOOST_CHECK(MAX_ITER != idx);
				BOOST_CHECK_EQUAL(client.isConnected(), false);
			}

			clientWorkerThread.reset();
			BOOST_TEST_MESSAGE("terminate thread for TCP socket client");
		}
		BOOST_TEST_MESSAGE("finish asynchronous TCP socket client that communicates with TCP socket server w/ session");
	}
};

struct TcpSocketClientTestSuite: public boost::unit_test_framework::test_suite
{
	TcpSocketClientTestSuite()
	: boost::unit_test_framework::test_suite("SWL.Util.TcpSocketClient")
	{
		boost::shared_ptr<EchoTcpSocketTest> test(new EchoTcpSocketTest());

		add(BOOST_CLASS_TEST_CASE(&EchoTcpSocketTest::testSyncClient, test), 0);
		add(BOOST_CLASS_TEST_CASE(&EchoTcpSocketTest::testAsyncClient, test), 0);
		//add(BOOST_TEST_CASE(boost::bind(&EchoTcpSocketTest::testSyncClient, test)), 0);
		//add(BOOST_TEST_CASE(boost::bind(&EchoTcpSocketTest::testAsyncClient, test)), 0);
		//add(BOOST_FIXTURE_TEST_CASE(boost::bind(&EchoTcpSocketTest::testSyncClient, test), Fixture), 0);  // not working
		//add(BOOST_FIXTURE_TEST_CASE(boost::bind(&EchoTcpSocketTest::testAsyncClient, test), Fixture), 0);  // not working

		boost::unit_test::framework::master_test_suite().add(this);
	}
} testsuite;

}  // unnamed namespace

//-----------------------------------------------------------------------------
//

#elif defined(__SWL_UNIT_TEST__USE_CPP_UNIT)

struct EchoTcpSocketTest: public CppUnit::TestFixture
{
private:
	CPPUNIT_TEST_SUITE(EchoTcpSocketTest);
	CPPUNIT_TEST(testSyncClient);
	CPPUNIT_TEST(testAsyncClient);
	CPPUNIT_TEST_SUITE_END();

private:
	boost::scoped_ptr<boost::thread> thrd_;

public:
	void setUp()  // set up
	{
		//CPPUNIT_MESSAGE("start thread for TCP socket servers");
		thrd_.reset(new boost::thread(echo_tcp_socket_server_worker_thread_functor()));
	}

	void tearDown()  // tear down
	{
		thrd_.reset();
		//CPPUNIT_MESSAGE("terminate thread for TCP socket servers");
	}

	void testSyncClient()
	{
		boost::asio::io_service ioService;
		swl::TcpSocketClient client(ioService);

		// synchronous TCP socket client that communicates with TCP socket server w/o session
		//CPPUNIT_MESSAGE("start synchronous TCP socket client that communicates with TCP socket server w/o session");
		{
#if defined(_UNICODE) || defined(UNICODE)
			std::wostringstream stream;
#else
			std::ostringstream stream;
#endif
			stream << serverPortNumber_withoutSession;
			const bool ret = client.connect(SWL_STR("localhost"), stream.str());
			CPPUNIT_ASSERT(ret);
			if (ret)
			{
				CPPUNIT_ASSERT_EQUAL(client.isConnected(), true);

				{
					const unsigned char sendMsg[] = "abcdefghijklmnopqrstuvwxyz";
					unsigned char receiveMsg[256] = { '\0', };
					const std::size_t sendingLen = std::strlen((char *)sendMsg) * sizeof(sendMsg[0]);
					const std::size_t maxReceivingLen = sizeof(receiveMsg) * sizeof(receiveMsg[0]);
					const size_t sentLen = client.send(sendMsg, sendingLen);
					CPPUNIT_ASSERT_EQUAL(sendingLen, sentLen);
					const size_t receivedLen = client.receive(receiveMsg, maxReceivingLen);
					CPPUNIT_ASSERT_EQUAL(std::strlen((char *)receiveMsg) * sizeof(receiveMsg[0]), receivedLen);
					CPPUNIT_ASSERT(std::strcmp((char *)receiveMsg, (char *)sendMsg) == 0);
				}

				{
					const unsigned char sendMsg[] = "9876543210";
					unsigned char receiveMsg[256] = { '\0', };
					const std::size_t sendingLen = std::strlen((char *)sendMsg) * sizeof(sendMsg[0]);
					const std::size_t maxReceivingLen = sizeof(receiveMsg) * sizeof(receiveMsg[0]);
					const size_t sentLen = client.send(sendMsg, sendingLen);
					CPPUNIT_ASSERT_EQUAL(sendingLen, sentLen);
					const size_t receivedLen = client.receive(receiveMsg, maxReceivingLen);
					CPPUNIT_ASSERT_EQUAL(std::strlen((char *)receiveMsg) * sizeof(receiveMsg[0]), receivedLen);
					CPPUNIT_ASSERT(std::strcmp((char *)receiveMsg, (char *)sendMsg) == 0);
				}

				client.disconnect();
				CPPUNIT_ASSERT_EQUAL(client.isConnected(), false);
			}
		}
		//CPPUNIT_MESSAGE("finish synchronous TCP socket client that communicates with TCP socket server w/o session");

		// synchronous TCP socket client that communicates with TCP socket server w/ session
		//CPPUNIT_MESSAGE("start synchronous TCP socket client that communicates with TCP socket server w/ session");
		{
#if defined(_UNICODE) || defined(UNICODE)
			std::wostringstream stream;
#else
			std::ostringstream stream;
#endif
			stream << serverPortNumber_withSession;
			const bool ret = client.connect(SWL_STR("localhost"), stream.str());
			CPPUNIT_ASSERT(ret);
			if (ret)
			{
				CPPUNIT_ASSERT_EQUAL(client.isConnected(), true);

				{
					const unsigned char sendMsg[] = "abcdefghijklmnopqrstuvwxyz";
					unsigned char receiveMsg[256] = { '\0', };
					const std::size_t sendingLen = std::strlen((char *)sendMsg) * sizeof(sendMsg[0]);
					const std::size_t maxReceivingLen = sizeof(receiveMsg) * sizeof(receiveMsg[0]);
					const size_t sentLen = client.send(sendMsg, sendingLen);
					CPPUNIT_ASSERT_EQUAL(sendingLen, sentLen);
					const size_t receivedLen = client.receive(receiveMsg, maxReceivingLen);
					CPPUNIT_ASSERT_EQUAL(std::strlen((char *)receiveMsg) * sizeof(receiveMsg[0]), receivedLen);
					CPPUNIT_ASSERT(std::strcmp((char *)receiveMsg, (char *)sendMsg) == 0);
				}

				{
					const unsigned char sendMsg[] = "9876543210";
					unsigned char receiveMsg[256] = { '\0', };
					const std::size_t sendingLen = std::strlen((char *)sendMsg) * sizeof(sendMsg[0]);
					const std::size_t maxReceivingLen = sizeof(receiveMsg) * sizeof(receiveMsg[0]);
					const size_t sentLen = client.send(sendMsg, sendingLen);
					CPPUNIT_ASSERT_EQUAL(sendingLen, sentLen);
					const size_t receivedLen = client.receive(receiveMsg, maxReceivingLen);
					CPPUNIT_ASSERT_EQUAL(std::strlen((char *)receiveMsg) * sizeof(receiveMsg[0]), receivedLen);
					CPPUNIT_ASSERT(std::strcmp((char *)receiveMsg, (char *)sendMsg) == 0);
				}

				client.disconnect();
				CPPUNIT_ASSERT_EQUAL(client.isConnected(), false);
			}
		}
		//CPPUNIT_MESSAGE("finish synchronous TCP socket client that communicates with TCP socket server w/ session");
	}

	void testAsyncClient()
	{
		const int MAX_ITER = 100;
		int idx = 0;

		// asynchronous TCP socket client that communicates with TCP socket server w/o session
		//CPPUNIT_MESSAGE("start asynchronous TCP socket client that communicates with TCP socket server w/o session");
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

			//CPPUNIT_MESSAGE("start thread for TCP socket client");
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
			CPPUNIT_ASSERT(MAX_ITER != idx);

			{
				CPPUNIT_ASSERT_EQUAL(client.isConnected(), true);
				CPPUNIT_ASSERT_EQUAL(client.isSendBufferEmpty(), true);
				CPPUNIT_ASSERT_EQUAL(client.isReceiveBufferEmpty(), true);

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
					CPPUNIT_ASSERT(MAX_ITER != idx);

					std::vector<unsigned char> msg(sendingLen);
					client.receive(&msg[0], sendingLen);
					CPPUNIT_ASSERT(!msg.empty());
					CPPUNIT_ASSERT(std::strncmp((char *)&msg[0], (char *)sendMsg, sendingLen) == 0);
				}

				CPPUNIT_ASSERT_EQUAL(client.isSendBufferEmpty(), true);
				CPPUNIT_ASSERT_EQUAL(client.isReceiveBufferEmpty(), true);
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
					CPPUNIT_ASSERT(MAX_ITER != idx);

					std::vector<unsigned char> msg(sendingLen);
					client.receive(&msg[0], sendingLen);
					CPPUNIT_ASSERT(!msg.empty());
					CPPUNIT_ASSERT(std::strncmp((char *)&msg[0], (char *)sendMsg, sendingLen) == 0);
				}

				CPPUNIT_ASSERT_EQUAL(client.isSendBufferEmpty(), true);
				CPPUNIT_ASSERT_EQUAL(client.isReceiveBufferEmpty(), true);

				client.disconnect();
				idx = 0;
				while (client.isConnected() && idx < MAX_ITER)
				{
					boost::asio::deadline_timer timer(ioService);
					timer.expires_from_now(boost::posix_time::milliseconds(10));
					timer.wait();
					++idx;
				}
				CPPUNIT_ASSERT(MAX_ITER != idx);
				CPPUNIT_ASSERT_EQUAL(client.isConnected(), false);
			}

			clientWorkerThread.reset();
			//CPPUNIT_MESSAGE("terminate thread for TCP socket client");
		}
		//CPPUNIT_MESSAGE("finish asynchronous TCP socket client that communicates with TCP socket server w/o session");

		// asynchronous TCP socket client that communicates with TCP socket server w/ session
		//CPPUNIT_MESSAGE("start asynchronous TCP socket client that communicates with TCP socket server w/ session");
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

			//CPPUNIT_MESSAGE("start thread for TCP socket client");
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
			CPPUNIT_ASSERT(MAX_ITER != idx);

			{
				CPPUNIT_ASSERT_EQUAL(client.isConnected(), true);
				CPPUNIT_ASSERT_EQUAL(client.isSendBufferEmpty(), true);
				CPPUNIT_ASSERT_EQUAL(client.isReceiveBufferEmpty(), true);

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
					CPPUNIT_ASSERT(MAX_ITER != idx);

					std::vector<unsigned char> msg(sendingLen);
					client.receive(&msg[0], sendingLen);
					CPPUNIT_ASSERT(!msg.empty());
					CPPUNIT_ASSERT(std::strncmp((char *)&msg[0], (char *)sendMsg, sendingLen) == 0);
				}

				CPPUNIT_ASSERT_EQUAL(client.isSendBufferEmpty(), true);
				CPPUNIT_ASSERT_EQUAL(client.isReceiveBufferEmpty(), true);
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
					CPPUNIT_ASSERT(MAX_ITER != idx);

					std::vector<unsigned char> msg(sendingLen);
					client.receive(&msg[0], sendingLen);
					CPPUNIT_ASSERT(!msg.empty());
					CPPUNIT_ASSERT(std::strncmp((char *)&msg[0], (char *)sendMsg, sendingLen) == 0);
				}

				CPPUNIT_ASSERT_EQUAL(client.isSendBufferEmpty(), true);
				CPPUNIT_ASSERT_EQUAL(client.isReceiveBufferEmpty(), true);

				client.disconnect();
				idx = 0;
				while (client.isConnected() && idx < MAX_ITER)
				{
					boost::asio::deadline_timer timer(ioService);
					timer.expires_from_now(boost::posix_time::milliseconds(10));
					timer.wait();
					++idx;
				}
				CPPUNIT_ASSERT(MAX_ITER != idx);
				CPPUNIT_ASSERT_EQUAL(client.isConnected(), false);
			}

			clientWorkerThread.reset();
			//CPPUNIT_MESSAGE("terminate thread for TCP socket client");
		}
		//CPPUNIT_MESSAGE("finish asynchronous TCP socket client that communicates with TCP socket server w/ session");
	}
};

#endif

}  // namespace unit_test
}  // namespace swl

#if defined(__SWL_UNIT_TEST__USE_CPP_UNIT)
CPPUNIT_TEST_SUITE_REGISTRATION(swl::unit_test::EchoTcpSocketTest);
//CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(swl::unit_test::EchoTcpSocketTest, "SWL.Util.TcpSocketClient");  // not working
#endif
