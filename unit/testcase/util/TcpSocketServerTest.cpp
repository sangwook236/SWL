#include "swl/Config.h"
#include "../../UnitTestConfig.h"
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

struct tcp_socket_worker_thread_functor
{
	void operator()()
	{
		boost::asio::io_service ioService;
		const unsigned short portNum_FullDuplex = 5005;
		const unsigned short portNum_HalfDuplex = 5006;

		TcpSocketServer<FullDuplexTcpSocketConnection<EchoTcpSocketSessionUsingFullDuplex> > fullDuplexServer(ioService, portNum_FullDuplex);
		TcpSocketServer<HalfDuplexTcpSocketConnection<EchoTcpSocketSessionUsingHalfDuplex> > halfDuplexServer(ioService, portNum_HalfDuplex);

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

struct TcpSocketServerTest
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
	void testTcpSocketServer()
	{
		Fixture fixture;

		BOOST_TEST_MESSAGE("TcpSocketServerTest::testTcpSocketServer() is called");
	}
};

struct TcpSocketServerTestSuite: public boost::unit_test_framework::test_suite
{
	TcpSocketServerTestSuite()
	: boost::unit_test_framework::test_suite("SWL.Util.TcpSocketServer")
	{
		boost::shared_ptr<TcpSocketServerTest> test(new TcpSocketServerTest());

		//add(BOOST_TEST_CASE(boost::bind(&TcpSocketServerTest::testTcpSocketServer, test)), 0);
		add(BOOST_CLASS_TEST_CASE(&TcpSocketServerTest::testTcpSocketServer, test), 0);
		//add(BOOST_FIXTURE_TEST_CASE(boost::bind(&TcpSocketServerTest::testTcpSocketServer, test), Fixture), 0);  // not working

		boost::unit_test::framework::master_test_suite().add(this);
	}
} testsuite;

}  // unnamed namespace

//-----------------------------------------------------------------------------
//

#elif defined(__SWL_UNIT_TEST__USE_CPP_UNIT)

struct TcpSocketServerTest: public CppUnit::TestFixture
{
private:
	CPPUNIT_TEST_SUITE(TcpSocketServerTest);
	CPPUNIT_TEST(testTcpSocketServer);
	CPPUNIT_TEST_SUITE_END();

public:
	void setUp()  // set up
	{
	}

	void tearDown()  // tear down
	{
	}

	void testTcpSocketServer()
	{
		boost::scoped_ptr<boost::thread> thrd(new boost::thread(tcp_socket_worker_thread_functor()));
		//if (thrd.get()) thrd->join();
	}
};

#endif

}  // namespace unit_test
}  // namespace swl

#if defined(__SWL_UNIT_TEST__USE_CPP_UNIT)
CPPUNIT_TEST_SUITE_REGISTRATION(swl::unit_test::TcpSocketServerTest);
//CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(swl::unit_test::TcpSocketServerTest, "SWL.Util.TcpSocketServer");  // not working
#endif
