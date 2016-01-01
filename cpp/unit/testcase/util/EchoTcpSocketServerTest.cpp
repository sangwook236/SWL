#include "swl/Config.h"
#include "../../UnitTestConfig.h"
#include "EchoTcpSocketConnection.h"
#include "EchoTcpSocketSession.h"
#include "swl/util/TcpSocketServer.h"
#include "swl/util/TcpSocketConnection.h"
#include "swl/util/TcpSocketConnectionUsingSession.h"
#include <boost/thread.hpp>
#include <boost/smart_ptr.hpp>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {
namespace unit_test {

struct echo_tcp_socket_server_worker_thread_functor
{
	void operator()()
	{
		boost::asio::io_service ioService;
		const unsigned short portNum_withoutSession = 6002;
		const unsigned short portNum_withSession = 7002;

		swl::TcpSocketServer<swl::EchoTcpSocketConnection> server(ioService, portNum_withoutSession);
		swl::TcpSocketServer<swl::TcpSocketConnectionUsingSession<swl::EchoTcpSocketSession> > sessionServer(ioService, portNum_withSession);

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

//-----------------------------------------------------------------------------
// Boost Test

#if defined(__SWL_UNIT_TEST__USE_BOOST_TEST)

namespace {

struct EchoTcpSocketServerTest
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
	void testServerRun()
	{
		Fixture fixture;

		BOOST_TEST_MESSAGE("start thread for TCP socket servers");
		boost::scoped_ptr<boost::thread> workerThread(new boost::thread(echo_tcp_socket_server_worker_thread_functor()));
		//if (workerThread.get()) workerThread->join();
		BOOST_TEST_MESSAGE("terminate thread for TCP socket servers");
	}
};

struct TcpSocketServerTestSuite: public boost::unit_test_framework::test_suite
{
	TcpSocketServerTestSuite()
	: boost::unit_test_framework::test_suite("SWL.Util.TcpSocketServer")
	{
		boost::shared_ptr<EchoTcpSocketServerTest> test(new EchoTcpSocketServerTest());

		add(BOOST_CLASS_TEST_CASE(&EchoTcpSocketServerTest::testServerRun, test), 0);
		//add(BOOST_TEST_CASE(boost::bind(&EchoTcpSocketServerTest::testServerRun, test)), 0);
		//add(BOOST_FIXTURE_TEST_CASE(boost::bind(&EchoTcpSocketServerTest::testServerRun, test), Fixture), 0);  // not working

		boost::unit_test::framework::master_test_suite().add(this);
	}
} testsuite;

}  // unnamed namespace

//-----------------------------------------------------------------------------
// Google Test

#elif defined(__SWL_UNIT_TEST__USE_GOOGLE_TEST)

class EchoTcpSocketServerTest : public testing::Test
{
protected:
	/*virtual*/ void SetUp()
	{
	}

	/*virtual*/ void TearDown()
	{
	}
};

TEST_F(EchoTcpSocketServerTest, testServerRun)
{
	// FIXME [fix] >> change SCOPED_TRACE to message output function.
	SCOPED_TRACE("start thread for TCP socket servers");
	boost::scoped_ptr<boost::thread> workerThread(new boost::thread(echo_tcp_socket_server_worker_thread_functor()));
	//if (workerThread.get()) workerThread->join();
	// FIXME [fix] >> change SCOPED_TRACE to message output function.
	SCOPED_TRACE("terminate thread for TCP socket servers");
}

//-----------------------------------------------------------------------------
// CppUnit

#elif defined(__SWL_UNIT_TEST__USE_CPP_UNIT)

struct EchoTcpSocketServerTest: public CppUnit::TestFixture
{
private:
	CPPUNIT_TEST_SUITE(EchoTcpSocketServerTest);
	CPPUNIT_TEST(testServerRun);
	CPPUNIT_TEST_SUITE_END();

public:
	void setUp()  // set up
	{
	}

	void tearDown()  // tear down
	{
	}

	void testServerRun()
	{
		//CPPUNIT_MESSAGE("start thread for TCP socket servers");
		boost::scoped_ptr<boost::thread> workerThread(new boost::thread(echo_tcp_socket_server_worker_thread_functor()));
		//if (workerThread.get()) workerThread->join();
		//CPPUNIT_MESSAGE("terminate thread for TCP socket servers");
	}
};

#endif

}  // namespace unit_test
}  // namespace swl

#if defined(__SWL_UNIT_TEST__USE_CPP_UNIT)
//CPPUNIT_TEST_SUITE_REGISTRATION(swl::unit_test::EchoTcpSocketServerTest);
CPPUNIT_REGISTRY_ADD_TO_DEFAULT("SWL.Util");
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(swl::unit_test::EchoTcpSocketServerTest, "SWL.Util");
#endif
