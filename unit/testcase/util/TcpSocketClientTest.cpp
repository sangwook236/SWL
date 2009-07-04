#include "../../UnitTestConfig.h"
#include "swl/util/TcpSocketClient.h"


#if defined(_MSC_VER) && defined(_DEBUG)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {
namespace unit_test {

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
	void test1()
	{
		Fixture fixture;

		BOOST_TEST_MESSAGE("TcpSocketClientTest::test1() is called");
	}

	void test2()
	{
		Fixture fixture;

		BOOST_TEST_MESSAGE("TcpSocketClientTest::test2() is called");
	}

	void test3()
	{
		Fixture fixture;

		BOOST_TEST_MESSAGE("TcpSocketClientTest::test3() is called");
	}
};

struct TcpSocketClientTestSuite: public boost::unit_test_framework::test_suite
{
	TcpSocketClientTestSuite()
	: boost::unit_test_framework::test_suite("SWL.Util.TcpSocketClient")
	{
		boost::shared_ptr<TcpSocketClientTest> test(new TcpSocketClientTest());

		//add(BOOST_TEST_CASE(boost::bind(&TcpSocketClientTest::test1, test)), 0);
		//add(BOOST_TEST_CASE(boost::bind(&TcpSocketClientTest::test2, test)), 0);
		//add(BOOST_TEST_CASE(boost::bind(&TcpSocketClientTest::test3, test)), 0);
		add(BOOST_CLASS_TEST_CASE(&TcpSocketClientTest::test1, test), 0);
		add(BOOST_CLASS_TEST_CASE(&TcpSocketClientTest::test2, test), 0);
		add(BOOST_CLASS_TEST_CASE(&TcpSocketClientTest::test3, test), 0);
		//add(BOOST_FIXTURE_TEST_CASE(boost::bind(&TcpSocketClientTest::test1, test), Fixture), 0);  // not working
		//add(BOOST_FIXTURE_TEST_CASE(boost::bind(&TcpSocketClientTest::test2, test), Fixture), 0);  // not working
		//add(BOOST_FIXTURE_TEST_CASE(boost::bind(&TcpSocketClientTest::test3, test), Fixture), 0);  // not working

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
	CPPUNIT_TEST(test1);
	CPPUNIT_TEST(test2);
	CPPUNIT_TEST(test3);
	CPPUNIT_TEST_SUITE_END();

public:
	void setUp()  // set up
	{
	}

	void tearDown()  // tear down
	{
	}

	void test1()
	{
	}

	void test2()
	{
	}

	void test3()
	{
	}
};

#endif

}  // namespace unit_test
}  // namespace swl

#if defined(__SWL_UNIT_TEST__USE_CPP_UNIT)
CPPUNIT_TEST_SUITE_REGISTRATION(swl::unit_test::TcpSocketClientTest);
//CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(swl::unit_test::TcpSocketClientTest, "SWL.Util.TcpSocketClient");  // not working
#endif
