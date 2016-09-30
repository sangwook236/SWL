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

#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {
namespace unit_test {

namespace {

const unsigned short serverPortNumber_withoutSession = 6001;
const unsigned short serverPortNumber_withSession = 7001;

struct echo_tcp_socket_server_worker_thread_functor
{
	void operator()()
	{
		boost::asio::io_service ioService;

		swl::TcpSocketServer<swl::TcpSocketConnectionUsingSession<swl::EchoTcpSocketSession> > sessionServer(ioService, serverPortNumber_withSession);
		swl::TcpSocketServer<swl::EchoTcpSocketConnection> server(ioService, serverPortNumber_withoutSession);

#if defined(__SWL_UNIT_TEST__USE_BOOST_TEST)
		BOOST_TEST_MESSAGE("start TCP socket servers: w/o & w/ session");
#elif defined(__SWL_UNIT_TEST__USE_GOOGLE_TEST)
		// FIXME [fix] >> change SCOPED_TRACE to message output function.
		SCOPED_TRACE("start TCP socket servers: w/o & w/ session");
#endif
		ioService.run();
#if defined(__SWL_UNIT_TEST__USE_BOOST_TEST)
		BOOST_TEST_MESSAGE("finish TCP socket servers: w/o & w/ session");
#elif defined(__SWL_UNIT_TEST__USE_GOOGLE_TEST)
		// FIXME [fix] >> change SCOPED_TRACE to message output function.
		SCOPED_TRACE("finish TCP socket servers: w/o & w/ session");
#endif
	}
};

}  // unnamed namespace

//-----------------------------------------------------------------------------
// Boost Test.

#if defined(__SWL_UNIT_TEST__USE_BOOST_TEST)

namespace {

struct EchoTcpSocketTest
{
private:
	struct Fixture
	{
		Fixture()  // Set up.
		{
			BOOST_TEST_MESSAGE("Start thread for TCP socket servers");
			thrd_.reset(new boost::thread(echo_tcp_socket_server_worker_thread_functor()));
		}

		~Fixture()  // Tear down.
		{
			thrd_.reset();
			BOOST_TEST_MESSAGE("Terminate thread for TCP socket servers");
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
				BOOST_CHECK(client.isConnected());

				{
					const unsigned char sendMsg[] = "abcdefghijklmnopqrstuvwxyz";
					unsigned char receiveMsg[256] = { '\0', };
					const std::size_t sendingLen = std::strlen((char *)sendMsg) * sizeof(sendMsg[0]);
					const std::size_t maxReceivingLen = sizeof(receiveMsg) * sizeof(receiveMsg[0]);
					const size_t sentLen = client.send(sendMsg, sendingLen);
					BOOST_CHECK_EQUAL(sendingLen, sentLen);
					const size_t receivedLen = client.receive(receiveMsg, maxReceivingLen);
					BOOST_CHECK_EQUAL(std::strlen((char *)receiveMsg) * sizeof(receiveMsg[0]), receivedLen);
					BOOST_CHECK_EQUAL(std::strcmp((char *)receiveMsg, (char *)sendMsg), 0);
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
					BOOST_CHECK_EQUAL(std::strcmp((char *)receiveMsg, (char *)sendMsg), 0);
				}

				client.disconnect();
				BOOST_CHECK(!client.isConnected());
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
				BOOST_CHECK(client.isConnected());

				{
					const unsigned char sendMsg[] = "abcdefghijklmnopqrstuvwxyz";
					unsigned char receiveMsg[256] = { '\0', };
					const std::size_t sendingLen = std::strlen((char *)sendMsg) * sizeof(sendMsg[0]);
					const std::size_t maxReceivingLen = sizeof(receiveMsg) * sizeof(receiveMsg[0]);
					const size_t sentLen = client.send(sendMsg, sendingLen);
					BOOST_CHECK_EQUAL(sendingLen, sentLen);
					const size_t receivedLen = client.receive(receiveMsg, maxReceivingLen);
					BOOST_CHECK_EQUAL(std::strlen((char *)receiveMsg) * sizeof(receiveMsg[0]), receivedLen);
					BOOST_CHECK_EQUAL(std::strcmp((char *)receiveMsg, (char *)sendMsg), 0);
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
					BOOST_CHECK_EQUAL(std::strcmp((char *)receiveMsg, (char *)sendMsg), 0);
				}

				client.disconnect();
				BOOST_CHECK(!client.isConnected());
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
			BOOST_CHECK_NE(MAX_ITER, idx);

			{
				BOOST_CHECK(client.isConnected());
				BOOST_CHECK(client.isSendBufferEmpty());
				BOOST_CHECK(client.isReceiveBufferEmpty());

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
					BOOST_CHECK_EQUAL(MAX_ITER, idx);

					std::vector<unsigned char> msg(sendingLen);
					client.receive(&msg[0], sendingLen);
					BOOST_CHECK(!msg.empty());
					BOOST_CHECK_EQUAL(std::strncmp((char *)&msg[0], (char *)sendMsg, sendingLen), 0);
				}

				BOOST_CHECK(client.isSendBufferEmpty());
				BOOST_CHECK(client.isReceiveBufferEmpty());
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
					BOOST_CHECK_EQUAL(MAX_ITER, idx);

					std::vector<unsigned char> msg(sendingLen);
					client.receive(&msg[0], sendingLen);
					BOOST_CHECK(!msg.empty());
					BOOST_CHECK_EQUAL(std::strncmp((char *)&msg[0], (char *)sendMsg, sendingLen), 0);
				}

				BOOST_CHECK(client.isSendBufferEmpty());
				BOOST_CHECK(client.isReceiveBufferEmpty());

				client.disconnect();
				idx = 0;
				while (client.isConnected() && idx < MAX_ITER)
				{
					boost::asio::deadline_timer timer(ioService);
					timer.expires_from_now(boost::posix_time::milliseconds(10));
					timer.wait();
					++idx;
				}
				BOOST_CHECK_NE(MAX_ITER, idx);
				BOOST_CHECK(!client.isConnected());
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
			BOOST_CHECK_NE(MAX_ITER, idx);

			{
				BOOST_CHECK(client.isConnected());
				BOOST_CHECK(client.isSendBufferEmpty());
				BOOST_CHECK(client.isReceiveBufferEmpty());

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
					BOOST_CHECK_NE(MAX_ITER, idx);

					std::vector<unsigned char> msg(sendingLen);
					client.receive(&msg[0], sendingLen);
					BOOST_CHECK(!msg.empty());
					BOOST_CHECK_EQUAL(std::strncmp((char *)&msg[0], (char *)sendMsg, sendingLen), 0);
				}

				BOOST_CHECK(client.isSendBufferEmpty());
				BOOST_CHECK(client.isReceiveBufferEmpty());
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
					BOOST_CHECK_NE(MAX_ITER, idx);

					std::vector<unsigned char> msg(sendingLen);
					client.receive(&msg[0], sendingLen);
					BOOST_CHECK(!msg.empty());
					BOOST_CHECK_EQUAL(std::strncmp((char *)&msg[0], (char *)sendMsg, sendingLen), 0);
				}

				BOOST_CHECK(client.isSendBufferEmpty());
				BOOST_CHECK(client.isReceiveBufferEmpty());

				client.disconnect();
				idx = 0;
				while (client.isConnected() && idx < MAX_ITER)
				{
					boost::asio::deadline_timer timer(ioService);
					timer.expires_from_now(boost::posix_time::milliseconds(10));
					timer.wait();
					++idx;
				}
				BOOST_CHECK_NE(MAX_ITER, idx);
				BOOST_CHECK(!client.isConnected());
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
// Google Test.

#elif defined(__SWL_UNIT_TEST__USE_GOOGLE_TEST)

class EchoTcpSocketTest : public testing::Test
{
protected:
	/*virtual*/ void SetUp()
	{
		// FIXME [fix] >> change SCOPED_TRACE to message output function.
		SCOPED_TRACE("Start thread for TCP socket servers");
		thrd_.reset(new boost::thread(echo_tcp_socket_server_worker_thread_functor()));
	}

	/*virtual*/ void TearDown()
	{
		thrd_.reset();
		// FIXME [fix] >> change SCOPED_TRACE to message output function.
		SCOPED_TRACE("Terminate thread for TCP socket servers");
	}

private:
	boost::scoped_ptr<boost::thread> thrd_;
};

TEST_F(EchoTcpSocketTest, testSyncClient)
{
	boost::asio::io_service ioService;
	swl::TcpSocketClient client(ioService);

	// synchronous TCP socket client that communicates with TCP socket server w/o session
	// FIXME [fix] >> change SCOPED_TRACE to message output function.
	SCOPED_TRACE("start synchronous TCP socket client that communicates with TCP socket server w/o session");
	{
#if defined(_UNICODE) || defined(UNICODE)
		std::wostringstream stream;
#else
		std::ostringstream stream;
#endif
		stream << serverPortNumber_withoutSession;
		const bool ret = client.connect(SWL_STR("localhost"), stream.str());
		EXPECT_TRUE(ret);
		if (ret)
		{
			EXPECT_TRUE(client.isConnected());

			{
				const unsigned char sendMsg[] = "abcdefghijklmnopqrstuvwxyz";
				unsigned char receiveMsg[256] = { '\0', };
				const std::size_t sendingLen = std::strlen((char *)sendMsg) * sizeof(sendMsg[0]);
				const std::size_t maxReceivingLen = sizeof(receiveMsg) * sizeof(receiveMsg[0]);
				const size_t sentLen = client.send(sendMsg, sendingLen);
				EXPECT_EQ(sendingLen, sentLen);
				const size_t receivedLen = client.receive(receiveMsg, maxReceivingLen);
				EXPECT_EQ(std::strlen((char *)receiveMsg) * sizeof(receiveMsg[0]), receivedLen);
				EXPECT_EQ(std::strcmp((char *)receiveMsg, (char *)sendMsg), 0);
			}

			{
				const unsigned char sendMsg[] = "9876543210";
				unsigned char receiveMsg[256] = { '\0', };
				const std::size_t sendingLen = std::strlen((char *)sendMsg) * sizeof(sendMsg[0]);
				const std::size_t maxReceivingLen = sizeof(receiveMsg) * sizeof(receiveMsg[0]);
				const size_t sentLen = client.send(sendMsg, sendingLen);
				EXPECT_EQ(sendingLen, sentLen);
				const size_t receivedLen = client.receive(receiveMsg, maxReceivingLen);
				EXPECT_EQ(std::strlen((char *)receiveMsg) * sizeof(receiveMsg[0]), receivedLen);
				EXPECT_EQ(std::strcmp((char *)receiveMsg, (char *)sendMsg), 0);
			}

			client.disconnect();
			EXPECT_TRUE(!client.isConnected());
		}
	}
	// FIXME [fix] >> change SCOPED_TRACE to message output function.
	SCOPED_TRACE("finish synchronous TCP socket client that communicates with TCP socket server w/o session");

	// synchronous TCP socket client that communicates with TCP socket server using session
	// FIXME [fix] >> change SCOPED_TRACE to message output function.
	SCOPED_TRACE("start synchronous TCP socket client that communicates with TCP socket server w/ session");
	{
#if defined(_UNICODE) || defined(UNICODE)
		std::wostringstream stream;
#else
		std::ostringstream stream;
#endif
		stream << serverPortNumber_withSession;
		const bool ret = client.connect(SWL_STR("localhost"), stream.str());
		EXPECT_TRUE(ret);
		if (ret)
		{
			EXPECT_TRUE(client.isConnected());

			{
				const unsigned char sendMsg[] = "abcdefghijklmnopqrstuvwxyz";
				unsigned char receiveMsg[256] = { '\0', };
				const std::size_t sendingLen = std::strlen((char *)sendMsg) * sizeof(sendMsg[0]);
				const std::size_t maxReceivingLen = sizeof(receiveMsg) * sizeof(receiveMsg[0]);
				const size_t sentLen = client.send(sendMsg, sendingLen);
				EXPECT_EQ(sendingLen, sentLen);
				const size_t receivedLen = client.receive(receiveMsg, maxReceivingLen);
				EXPECT_EQ(std::strlen((char *)receiveMsg) * sizeof(receiveMsg[0]), receivedLen);
				EXPECT_EQ(std::strcmp((char *)receiveMsg, (char *)sendMsg), 0);
			}

			{
				const unsigned char sendMsg[] = "9876543210";
				unsigned char receiveMsg[256] = { '\0', };
				const std::size_t sendingLen = std::strlen((char *)sendMsg) * sizeof(sendMsg[0]);
				const std::size_t maxReceivingLen = sizeof(receiveMsg) * sizeof(receiveMsg[0]);
				const size_t sentLen = client.send(sendMsg, sendingLen);
				EXPECT_EQ(sendingLen, sentLen);
				const size_t receivedLen = client.receive(receiveMsg, maxReceivingLen);
				EXPECT_EQ(std::strlen((char *)receiveMsg) * sizeof(receiveMsg[0]), receivedLen);
				EXPECT_EQ(std::strcmp((char *)receiveMsg, (char *)sendMsg), 0);
			}

			client.disconnect();
			EXPECT_TRUE(!client.isConnected());
		}
	}
	// FIXME [fix] >> change SCOPED_TRACE to message output function.
	SCOPED_TRACE("finish synchronous TCP socket client that communicates with TCP socket server w/ session");
}

TEST_F(EchoTcpSocketTest, testAsyncClient)
{
	const int MAX_ITER = 100;
	int idx = 0;

	// asynchronous TCP socket client that communicates with TCP socket server w/o session
	// FIXME [fix] >> change SCOPED_TRACE to message output function.
	SCOPED_TRACE("start asynchronous TCP socket client that communicates with TCP socket server w/o session");
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

		// FIXME [fix] >> change SCOPED_TRACE to message output function.
		SCOPED_TRACE("start thread for TCP socket client");
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
		EXPECT_NE(MAX_ITER, idx);

		{
			EXPECT_TRUE(client.isConnected());
			EXPECT_TRUE(client.isSendBufferEmpty());
			EXPECT_TRUE(client.isReceiveBufferEmpty());

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
				EXPECT_EQ(MAX_ITER, idx);

				std::vector<unsigned char> msg(sendingLen);
				client.receive(&msg[0], sendingLen);
				EXPECT_FALSE(msg.empty());
				EXPECT_EQ(std::strncmp((char *)&msg[0], (char *)sendMsg, sendingLen), 0);
			}

			EXPECT_TRUE(client.isSendBufferEmpty());
			EXPECT_TRUE(client.isReceiveBufferEmpty());
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
				EXPECT_EQ(MAX_ITER, idx);

				std::vector<unsigned char> msg(sendingLen);
				client.receive(&msg[0], sendingLen);
				EXPECT_FALSE(msg.empty());
				EXPECT_EQ(std::strncmp((char *)&msg[0], (char *)sendMsg, sendingLen), 0);
			}

			EXPECT_TRUE(client.isSendBufferEmpty());
			EXPECT_TRUE(client.isReceiveBufferEmpty());

			client.disconnect();
			idx = 0;
			while (client.isConnected() && idx < MAX_ITER)
			{
				boost::asio::deadline_timer timer(ioService);
				timer.expires_from_now(boost::posix_time::milliseconds(10));
				timer.wait();
				++idx;
			}
			EXPECT_NE(MAX_ITER, idx);
			EXPECT_FALSE(client.isConnected());
		}

		clientWorkerThread.reset();
		// FIXME [fix] >> change SCOPED_TRACE to message output function.
		SCOPED_TRACE("terminate thread for TCP socket client");
	}
	// FIXME [fix] >> change SCOPED_TRACE to message output function.
	SCOPED_TRACE("finish asynchronous TCP socket client that communicates with TCP socket server w/o session");

	// asynchronous TCP socket client that communicates with TCP socket server using session
	// FIXME [fix] >> change SCOPED_TRACE to message output function.
	SCOPED_TRACE("start asynchronous TCP socket client that communicates with TCP socket server w/ session");
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

		// FIXME [fix] >> change SCOPED_TRACE to message output function.
		SCOPED_TRACE("start thread for TCP socket client");
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
		EXPECT_NE(MAX_ITER, idx);

		{
			EXPECT_TRUE(client.isConnected());
			EXPECT_TRUE(client.isSendBufferEmpty());
			EXPECT_TRUE(client.isReceiveBufferEmpty());

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
				EXPECT_NE(MAX_ITER, idx);

				std::vector<unsigned char> msg(sendingLen);
				client.receive(&msg[0], sendingLen);
				EXPECT_FALSE(msg.empty());
				EXPECT_EQ(std::strncmp((char *)&msg[0], (char *)sendMsg, sendingLen), 0);
			}

			EXPECT_TRUE(client.isSendBufferEmpty());
			EXPECT_TRUE(client.isReceiveBufferEmpty());
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
				EXPECT_NE(MAX_ITER, idx);

				std::vector<unsigned char> msg(sendingLen);
				client.receive(&msg[0], sendingLen);
				EXPECT_FALSE(msg.empty());
				EXPECT_EQ(std::strncmp((char *)&msg[0], (char *)sendMsg, sendingLen), 0);
			}

			EXPECT_TRUE(client.isSendBufferEmpty());
			EXPECT_TRUE(client.isReceiveBufferEmpty());

			client.disconnect();
			idx = 0;
			while (client.isConnected() && idx < MAX_ITER)
			{
				boost::asio::deadline_timer timer(ioService);
				timer.expires_from_now(boost::posix_time::milliseconds(10));
				timer.wait();
				++idx;
			}
			EXPECT_NE(MAX_ITER, idx);
			EXPECT_FALSE(client.isConnected());
		}

		clientWorkerThread.reset();
		// FIXME [fix] >> change SCOPED_TRACE to message output function.
		SCOPED_TRACE("terminate thread for TCP socket client");
	}
	// FIXME [fix] >> change SCOPED_TRACE to message output function.
	SCOPED_TRACE("finish asynchronous TCP socket client that communicates with TCP socket server w/ session");
}

//-----------------------------------------------------------------------------
// CppUnit.

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
	void setUp()  // Set up.
	{
		//CPPUNIT_MESSAGE("Start thread for TCP socket servers");
		thrd_.reset(new boost::thread(echo_tcp_socket_server_worker_thread_functor()));
	}

	void tearDown()  // Tear down.
	{
		thrd_.reset();
		//CPPUNIT_MESSAGE("Terminate thread for TCP socket servers");
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
				CPPUNIT_ASSERT(client.isConnected());

				{
					const unsigned char sendMsg[] = "abcdefghijklmnopqrstuvwxyz";
					unsigned char receiveMsg[256] = { '\0', };
					const std::size_t sendingLen = std::strlen((char *)sendMsg) * sizeof(sendMsg[0]);
					const std::size_t maxReceivingLen = sizeof(receiveMsg) * sizeof(receiveMsg[0]);
					const size_t sentLen = client.send(sendMsg, sendingLen);
					CPPUNIT_ASSERT_EQUAL(sendingLen, sentLen);
					const size_t receivedLen = client.receive(receiveMsg, maxReceivingLen);
					CPPUNIT_ASSERT_EQUAL(std::strlen((char *)receiveMsg) * sizeof(receiveMsg[0]), receivedLen);
					CPPUNIT_ASSERT_EQUAL(std::strcmp((char *)receiveMsg, (char *)sendMsg), 0);
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
					CPPUNIT_ASSERT_EQUAL(std::strcmp((char *)receiveMsg, (char *)sendMsg), 0);
				}

				client.disconnect();
				CPPUNIT_ASSERT(!client.isConnected());
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
				CPPUNIT_ASSERT(client.isConnected());

				{
					const unsigned char sendMsg[] = "abcdefghijklmnopqrstuvwxyz";
					unsigned char receiveMsg[256] = { '\0', };
					const std::size_t sendingLen = std::strlen((char *)sendMsg) * sizeof(sendMsg[0]);
					const std::size_t maxReceivingLen = sizeof(receiveMsg) * sizeof(receiveMsg[0]);
					const size_t sentLen = client.send(sendMsg, sendingLen);
					CPPUNIT_ASSERT_EQUAL(sendingLen, sentLen);
					const size_t receivedLen = client.receive(receiveMsg, maxReceivingLen);
					CPPUNIT_ASSERT_EQUAL(std::strlen((char *)receiveMsg) * sizeof(receiveMsg[0]), receivedLen);
					CPPUNIT_ASSERT_EQUAL(std::strcmp((char *)receiveMsg, (char *)sendMsg), 0);
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
					CPPUNIT_ASSERT_EQUAL(std::strcmp((char *)receiveMsg, (char *)sendMsg), 0);
				}

				client.disconnect();
				CPPUNIT_ASSERT(!client.isConnected());
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
				timer.expires_from_now(boost::posix_time::milliseconds(100));
				timer.wait();
				++idx;
			}
			CPPUNIT_ASSERT(MAX_ITER != idx);

			{
				CPPUNIT_ASSERT(client.isConnected());
				CPPUNIT_ASSERT(client.isSendBufferEmpty());
				CPPUNIT_ASSERT(client.isReceiveBufferEmpty());

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
					CPPUNIT_ASSERT_EQUAL(std::strncmp((char *)&msg[0], (char *)sendMsg, sendingLen), 0);
				}

				CPPUNIT_ASSERT(client.isSendBufferEmpty());
				CPPUNIT_ASSERT(client.isReceiveBufferEmpty());
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
					CPPUNIT_ASSERT_EQUAL(std::strncmp((char *)&msg[0], (char *)sendMsg, sendingLen), 0);
				}

				CPPUNIT_ASSERT(client.isSendBufferEmpty());
				CPPUNIT_ASSERT(client.isReceiveBufferEmpty());

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
				CPPUNIT_ASSERT(!client.isConnected());
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
				timer.expires_from_now(boost::posix_time::milliseconds(100));
				timer.wait();
				++idx;
			}
			CPPUNIT_ASSERT(MAX_ITER != idx);

			{
				CPPUNIT_ASSERT(client.isConnected());
				CPPUNIT_ASSERT(client.isSendBufferEmpty());
				CPPUNIT_ASSERT(client.isReceiveBufferEmpty());

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
					CPPUNIT_ASSERT_EQUAL(std::strncmp((char *)&msg[0], (char *)sendMsg, sendingLen), 0);
				}

				CPPUNIT_ASSERT(client.isSendBufferEmpty());
				CPPUNIT_ASSERT(client.isReceiveBufferEmpty());
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
					CPPUNIT_ASSERT_EQUAL(std::strncmp((char *)&msg[0], (char *)sendMsg, sendingLen), 0);
				}

				CPPUNIT_ASSERT(client.isSendBufferEmpty());
				CPPUNIT_ASSERT(client.isReceiveBufferEmpty());

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
				CPPUNIT_ASSERT(!client.isConnected());
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
//CPPUNIT_TEST_SUITE_REGISTRATION(swl::unit_test::EchoTcpSocketTest);
CPPUNIT_REGISTRY_ADD_TO_DEFAULT("SWL.Util");
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(swl::unit_test::EchoTcpSocketTest, "SWL.Util");
#endif
