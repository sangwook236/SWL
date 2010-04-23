#include "swl/Config.h"
#include "../../UnitTestConfig.h"
#include "swl/base/String.h"
#include "swl/math/Line.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {
namespace unit_test {

//-----------------------------------------------------------------------------
//

#if defined(__SWL_UNIT_TEST__USE_BOOST_UNIT)

namespace {

struct Line3Test
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
	void test()
	{
		Fixture fixture;

	}
};

struct LineTestSuite: public boost::unit_test_framework::test_suite
{
	LineTestSuite()
	: boost::unit_test_framework::test_suite("SWL.Math.Line")
	{
		boost::shared_ptr<Line3Test> test(new Line3Test());

		add(BOOST_CLASS_TEST_CASE(&Line3Test::test, test), 0);

		boost::unit_test::framework::master_test_suite().add(this);
	}
} testsuite;

}  // unnamed namespace

//-----------------------------------------------------------------------------
//

#elif defined(__SWL_UNIT_TEST__USE_CPP_UNIT)

struct Line3Test: public CppUnit::TestFixture
{
private:
	CPPUNIT_TEST_SUITE(Line3Test);
	CPPUNIT_TEST(test);
	CPPUNIT_TEST_SUITE_END();

public:
	void setUp()  // set up
	{
	}

	void tearDown()  // tear down
	{
	}

	void test()
	{
	}
};

#endif

}  // namespace unit_test
}  // namespace swl

#if defined(__SWL_UNIT_TEST__USE_CPP_UNIT)
CPPUNIT_TEST_SUITE_REGISTRATION(swl::unit_test::Line3Test);
//CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(swl::unit_test::Line3Test, "SWL.Math.Line");  // not working
#endif
