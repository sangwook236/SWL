#include "swl/Config.h"
#include "../../UnitTestConfig.h"
#include "swl/util/TcpSocketClient.h"
#include "swl/base/String.h"
#include <sstream>

#include "EchoTcpSocketSessionUsingFullDuplex.h"
#include "EchoTcpSocketSessionUsingHalfDuplex.h"
#include "swl/util/TcpSocketServer.h"
#include "swl/util/TcpSocketConnection.h"
#include <boost/thread.hpp>
#include <boost/smart_ptr.hpp>

#if defined(_DEBUG)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {
namespace unit_test {

namespace {

const unsigned short FullDuplexServerPortNumer = 10002;
const unsigned short HalfDuplexServerPortNumer = 10003;

}  // unnamed namespace

struct tcp_socket_worker_thread_functor
{
	void operator()()
	{
		boost::asio::io_service ioService;

		TcpSocketServer<FullDuplexTcpSocketConnection<EchoTcpSocketSessionUsingFullDuplex> > fullDuplexServer(ioService, FullDuplexServerPortNumer);
		TcpSocketServer<HalfDuplexTcpSocketConnection<EchoTcpSocketSessionUsingHalfDuplex> > halfDuplexServer(ioService, HalfDuplexServerPortNumer);

#if defined(__SWL_UNIT_TEST__USE_BOOST_UNIT)
		BOOST_TEST_MESSAGE("server is started");
#endif
		ioService.run();
#if defined(__SWL_UNIT_TEST__USE_BOOST_UNIT)
		BOOST_TEST_MESSAGE("server is terminated");
#endif
	}
};

//-----------------------------------------------------------------------------
//

#if defined(__SWL_UNIT_TEST__USE_BOOST_UNIT)

namespace {

struct TcpSocketClientTest
{
private:
	struct Fixture
	{
		Fixture()  // set up
		{
		}

		~Fixture()  // tear down
		{
		}
	};

public:
	void testAsyncTcpSocketClient()
	{
		Fixture fixture;

		BOOST_TEST_MESSAGE("TcpSocketClientTest::testAsyncTcpSocketClient() is called");
	}

	void testSyncTcpSocketClient()
	{
		boost::scoped_ptr<boost::thread> thrd(new boost::thread(tcp_socket_worker_thread_functor()));

		boost::asio::io_service ioService;
		TcpSocketClient client(ioService);

		// full duplex TCP socket server
#if defined(_UNICODE) || defined(UNICODE)
		std::wostringstream stream1;
#else
		std::ostringstream stream1;
#endif
		stream1 << FullDuplexServerPortNumer;
		const bool ret1 = client.connect(SWL_STR("localhost"), stream1.str());
		BOOST_CHECK_EQUAL(client.isConnected(), ret1);
		BOOST_CHECK(ret1);
		if (ret1)
		{
			const unsigned char sendMsg1[] = "abcdefghijklmnopqrstuvwxyz";
			unsigned char receiveMsg1[256] = { '\0', };
			const size_t sendLen1 = client.send(sendMsg1, std::strlen((char *)sendMsg1) * sizeof(sendMsg1[0]));
			BOOST_CHECK_EQUAL(std::strlen((char *)sendMsg1) * sizeof(sendMsg1[0]), sendLen1);
			const size_t receiveLen1 = client.receive(receiveMsg1, sizeof(receiveMsg1) * sizeof(receiveMsg1[0]));
			BOOST_CHECK_EQUAL(std::strlen((char *)receiveMsg1) * sizeof(sendMsg1[0]), receiveLen1);
			BOOST_CHECK(std::strcmp((char *)receiveMsg1, (char *)sendMsg1) == 0);

			const unsigned char sendMsg2[] = "9876543210";
			unsigned char receiveMsg2[256] = { '\0', };
			const size_t sendLen2 = client.send(sendMsg2, std::strlen((char *)sendMsg2) * sizeof(sendMsg2[0]));
			BOOST_CHECK_EQUAL(std::strlen((char *)sendMsg2) * sizeof(sendMsg1[0]), sendLen2);
			const size_t receiveLen2 = client.receive(receiveMsg2, sizeof(receiveMsg2) * sizeof(receiveMsg2[0]));
			BOOST_CHECK_EQUAL(std::strlen((char *)receiveMsg2) * sizeof(sendMsg1[0]), receiveLen2);
			BOOST_CHECK(std::strcmp((char *)receiveMsg2, (char *)sendMsg2) == 0);

			client.disconnect();
			BOOST_CHECK_EQUAL(client.isConnected(), false);
		}

		// half duplex TCP socket server
#if defined(_UNICODE) || defined(UNICODE)
		std::wostringstream stream2;
#else
		std::ostringstream stream2;
#endif
		stream2 << HalfDuplexServerPortNumer;
		const bool ret2 = client.connect(SWL_STR("localhost"), stream2.str());
		BOOST_CHECK_EQUAL(client.isConnected(), ret2);
		BOOST_CHECK(ret2);
		if (ret2)
		{
			const unsigned char sendMsg1[] = "abcdefghijklmnopqrstuvwxyz";
			unsigned char receiveMsg1[256] = { '\0', };
			const size_t sendLen1 = client.send(sendMsg1, std::strlen((char *)sendMsg1) * sizeof(sendMsg1[0]));
			BOOST_CHECK_EQUAL(std::strlen((char *)sendMsg1) * sizeof(sendMsg1[0]), sendLen1);
			const size_t receiveLen1 = client.receive(receiveMsg1, sizeof(receiveMsg1) * sizeof(receiveMsg1[0]));
			BOOST_CHECK_EQUAL(std::strlen((char *)receiveMsg1) * sizeof(sendMsg1[0]), receiveLen1);
			BOOST_CHECK(std::strcmp((char *)receiveMsg1, (char *)sendMsg1) == 0);

			const unsigned char sendMsg2[] = "9876543210";
			unsigned char receiveMsg2[256] = { '\0', };
			const size_t sendLen2 = client.send(sendMsg2, std::strlen((char *)sendMsg2) * sizeof(sendMsg2[0]));
			BOOST_CHECK_EQUAL(std::strlen((char *)sendMsg2) * sizeof(sendMsg1[0]), sendLen2);
			const size_t receiveLen2 = client.receive(receiveMsg2, sizeof(receiveMsg2) * sizeof(receiveMsg2[0]));
			BOOST_CHECK_EQUAL(std::strlen((char *)receiveMsg2) * sizeof(sendMsg1[0]), receiveLen2);
			BOOST_CHECK(std::strcmp((char *)receiveMsg2, (char *)sendMsg2) == 0);

			client.disconnect();
			BOOST_CHECK_EQUAL(client.isConnected(), false);
		}
	}
};

struct TcpSocketClientTestSuite: public boost::unit_test_framework::test_suite
{
	TcpSocketClientTestSuite()
	: boost::unit_test_framework::test_suite("SWL.Util.TcpSocketClient")
	{
		boost::shared_ptr<TcpSocketClientTest> test(new TcpSocketClientTest());

		//add(BOOST_TEST_CASE(boost::bind(&TcpSocketClientTest::testAsyncTcpSocketClient, test)), 0);
		//add(BOOST_TEST_CASE(boost::bind(&TcpSocketClientTest::testSyncTcpSocketClient, test)), 0);
		add(BOOST_CLASS_TEST_CASE(&TcpSocketClientTest::testAsyncTcpSocketClient, test), 0);
		add(BOOST_CLASS_TEST_CASE(&TcpSocketClientTest::testSyncTcpSocketClient, test), 0);
		//add(BOOST_FIXTURE_TEST_CASE(boost::bind(&TcpSocketClientTest::testAsyncTcpSocketClient, test), Fixture), 0);  // not working
		//add(BOOST_FIXTURE_TEST_CASE(boost::bind(&TcpSocketClientTest::testSyncTcpSocketClient, test), Fixture), 0);  // not working

		boost::unit_test::framework::master_test_suite().add(this);
	}
} testsuite;

}  // unnamed namespace

//-----------------------------------------------------------------------------
//

#elif defined(__SWL_UNIT_TEST__USE_CPP_UNIT)

struct TcpSocketClientTest: public CppUnit::TestFixture
{
private:
	CPPUNIT_TEST_SUITE(TcpSocketClientTest);
	CPPUNIT_TEST(testAsyncTcpSocketClient);
	CPPUNIT_TEST(testSyncTcpSocketClient);
	CPPUNIT_TEST_SUITE_END();

public:
	void setUp()  // set up
	{
	}

	void tearDown()  // tear down
	{
	}

	void testAsyncTcpSocketClient()
	{
		boost::asio::io_service ioService;
		AsyncTcpSocketClient client(ioService);
	}

	void testSyncTcpSocketClient()
	{
		boost::scoped_ptr<boost::thread> thrd(new boost::thread(tcp_socket_worker_thread_functor()));
		
		boost::asio::io_service ioService;
		TcpSocketClient client(ioService);

		// full duplex TCP socket server
#if defined(_UNICODE) || defined(UNICODE)
		std::wostringstream stream1;
#else
		std::ostringstream stream1;
#endif
		stream1 << FullDuplexServerPortNumer;
		const bool ret1 = client.connect(SWL_STR("localhost"), stream1.str());
		CPPUNIT_ASSERT_EQUAL(client.isConnected(), ret1);
		CPPUNIT_ASSERT(ret1);
		if (ret1)
		{
			const unsigned char sendMsg1[] = "abcdefghijklmnopqrstuvwxyz";
			unsigned char receiveMsg1[256] = { '\0', };
			const size_t sendLen1 = client.send(sendMsg1, std::strlen((char *)sendMsg1) * sizeof(sendMsg1[0]));
			CPPUNIT_ASSERT_EQUAL(std::strlen((char *)sendMsg1) * sizeof(sendMsg1[0]), sendLen1);
			const size_t receiveLen1 = client.receive(receiveMsg1, sizeof(receiveMsg1) * sizeof(receiveMsg1[0]));
			CPPUNIT_ASSERT_EQUAL(std::strlen((char *)receiveMsg1) * sizeof(sendMsg1[0]), receiveLen1);
			CPPUNIT_ASSERT(std::strcmp((char *)receiveMsg1, (char *)sendMsg1) == 0);

			const unsigned char sendMsg2[] = "9876543210";
			unsigned char receiveMsg2[256] = { '\0', };
			const size_t sendLen2 = client.send(sendMsg2, std::strlen((char *)sendMsg2) * sizeof(sendMsg2[0]));
			CPPUNIT_ASSERT_EQUAL(std::strlen((char *)sendMsg2) * sizeof(sendMsg1[0]), sendLen2);
			const size_t receiveLen2 = client.receive(receiveMsg2, sizeof(receiveMsg2) * sizeof(receiveMsg2[0]));
			CPPUNIT_ASSERT_EQUAL(std::strlen((char *)receiveMsg2) * sizeof(sendMsg1[0]), receiveLen2);
			CPPUNIT_ASSERT(std::strcmp((char *)receiveMsg2, (char *)sendMsg2) == 0);

			client.disconnect();
			CPPUNIT_ASSERT_EQUAL(client.isConnected(), false);
		}

		// half duplex TCP socket server
#if defined(_UNICODE) || defined(UNICODE)
		std::wostringstream stream2;
#else
		std::ostringstream stream2;
#endif
		stream2 << HalfDuplexServerPortNumer;
		const bool ret2 = client.connect(SWL_STR("localhost"), stream2.str());
		CPPUNIT_ASSERT_EQUAL(client.isConnected(), ret2);
		CPPUNIT_ASSERT(ret2);
		if (ret2)
		{
			const unsigned char sendMsg1[] = "abcdefghijklmnopqrstuvwxyz";
			unsigned char receiveMsg1[256] = { '\0', };
			const size_t sendLen1 = client.send(sendMsg1, std::strlen((char *)sendMsg1) * sizeof(sendMsg1[0]));
			CPPUNIT_ASSERT_EQUAL(std::strlen((char *)sendMsg1) * sizeof(sendMsg1[0]), sendLen1);
			const size_t receiveLen1 = client.receive(receiveMsg1, sizeof(receiveMsg1) * sizeof(receiveMsg1[0]));
			CPPUNIT_ASSERT_EQUAL(std::strlen((char *)receiveMsg1) * sizeof(sendMsg1[0]), receiveLen1);
			CPPUNIT_ASSERT(std::strcmp((char *)receiveMsg1, (char *)sendMsg1) == 0);

			const unsigned char sendMsg2[] = "9876543210";
			unsigned char receiveMsg2[256] = { '\0', };
			const size_t sendLen2 = client.send(sendMsg2, std::strlen((char *)sendMsg2) * sizeof(sendMsg2[0]));
			CPPUNIT_ASSERT_EQUAL(std::strlen((char *)sendMsg2) * sizeof(sendMsg1[0]), sendLen2);
			const size_t receiveLen2 = client.receive(receiveMsg2, sizeof(receiveMsg2) * sizeof(receiveMsg2[0]));
			CPPUNIT_ASSERT_EQUAL(std::strlen((char *)receiveMsg2) * sizeof(sendMsg1[0]), receiveLen2);
			CPPUNIT_ASSERT(std::strcmp((char *)receiveMsg2, (char *)sendMsg2) == 0);

			client.disconnect();
			CPPUNIT_ASSERT_EQUAL(client.isConnected(), false);
		}
	}
};

#endif

}  // namespace unit_test
}  // namespace swl

#if defined(__SWL_UNIT_TEST__USE_CPP_UNIT)
CPPUNIT_TEST_SUITE_REGISTRATION(swl::unit_test::TcpSocketClientTest);
//CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(swl::unit_test::TcpSocketClientTest, "SWL.Util.TcpSocketClient");  // not working
#endif
