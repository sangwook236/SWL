#include "swl/Config.h"
#include "../../UnitTestConfig.h"

//#define __SWL__DISABLE_LOG_ 1
#include "swl/base/Log.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {
namespace unit_test {

//-----------------------------------------------------------------------------
// Boost Test.

#if defined(__SWL_UNIT_TEST__USE_BOOST_TEST)

namespace {

struct LogTest
{
private:
	struct Fixture
	{
		Fixture()  // Set up.
		{
		}

		~Fixture()  // Tear down.
		{
		}
	};

public:
	void testBasic()
	{
		Fixture fixture;

		SWL_LOG_WARN(L"Low fuel level.");
		SWL_LOG_ERROR(L"Located nearest gas station.");

		SWL_LOG_DEBUG(L"Starting search for nearest gas station.");
		SWL_LOG_DEBUG(L"Exiting gas station search.");
	}

public:
	LogTest()
	{
	}

private:
};

struct LogTestSuite: public boost::unit_test_framework::test_suite
{
	LogTestSuite()
	: boost::unit_test_framework::test_suite("SWL.Base.Log")
	{
		boost::shared_ptr<LogTest> test(new LogTest());

		add(BOOST_CLASS_TEST_CASE(&LogTest::testBasic, test), 0);

		boost::unit_test::framework::master_test_suite().add(this);
	}
} testsuite;

}  // unnamed namespace

//-----------------------------------------------------------------------------
// Google Test.

#elif defined(__SWL_UNIT_TEST__USE_GOOGLE_TEST)

class LogTest : public testing::Test
{
public:
	LogTest()
	{
	}

protected:
	/*virtual*/ void SetUp()
	{
	}

	/*virtual*/ void TearDown()
	{
	}
};

TEST_F(LogTest, testBasic)
{
	SWL_LOG_WARN(L"Low fuel level.");
	SWL_LOG_ERROR(L"Located nearest gas station.");

	SWL_LOG_DEBUG(L"Starting search for nearest gas station.");
	SWL_LOG_DEBUG(L"Exiting gas station search.");
}

//-----------------------------------------------------------------------------
// CppUnit.

#elif defined(__SWL_UNIT_TEST__USE_CPP_UNIT)

struct LogTest: public CppUnit::TestFixture
{
private:
	CPPUNIT_TEST_SUITE(LogTest);
	CPPUNIT_TEST(testBasic);
	CPPUNIT_TEST_SUITE_END();

public:
	void setUp()  // Set up.
	{
	}

	void tearDown()  // Tear down.
	{
	}

	void testBasic()
	{
		SWL_LOG_WARN(L"Low fuel level.");
		SWL_LOG_ERROR(L"Located nearest gas station.");

		SWL_LOG_DEBUG(L"Starting search for nearest gas station.");
		SWL_LOG_DEBUG(L"Exiting gas station search.");
	}

public:
	LogTest()
	{
	}
};

#endif

}  // namespace unit_test
}  // namespace swl

#if defined(__SWL_UNIT_TEST__USE_CPP_UNIT)
//CPPUNIT_TEST_SUITE_REGISTRATION(swl::unit_test::LogTest);
CPPUNIT_REGISTRY_ADD_TO_DEFAULT("SWL.Base");
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(swl::unit_test::LogTest, "SWL.Base");
#endif
