#include "swl/Config.h"
#include "../../UnitTestConfig.h"
#include "swl/base/String.h"
#include "swl/math/LineSegment.h"
#include <stdexcept>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {
namespace unit_test {

//-----------------------------------------------------------------------------
// Boost Test

#if defined(__SWL_UNIT_TEST__USE_BOOST_TEST)

namespace {

struct LineSegment3Test
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
	void testFoo()
	{
		Fixture fixture;

		throw std::runtime_error("not yet implemented");
	}
};

struct LineSegmentTestSuite: public boost::unit_test_framework::test_suite
{
	LineSegmentTestSuite()
	: boost::unit_test_framework::test_suite("SWL.Math.LineSegment")
	{
		boost::shared_ptr<LineSegment3Test> test(new LineSegment3Test());

		add(BOOST_CLASS_TEST_CASE(&LineSegment3Test::testFoo, test), 0);

		boost::unit_test::framework::master_test_suite().add(this);
	}
} testsuite;

}  // unnamed namespace

//-----------------------------------------------------------------------------
// Google Test

#elif defined(__SWL_UNIT_TEST__USE_GOOGLE_TEST)

class LineSegment3Test : public testing::Test
{
protected:
	/*virtual*/ void SetUp()
	{
	}

	/*virtual*/ void TearDown()
	{
	}
};

//TEST_F(LineSegment3Test, testFoo)
//{
//	throw std::runtime_error("not yet implemented");
//}

//-----------------------------------------------------------------------------
// CppUnit

#elif defined(__SWL_UNIT_TEST__USE_CPP_UNIT)

struct LineSegment3Test: public CppUnit::TestFixture
{
private:
	CPPUNIT_TEST_SUITE(LineSegment3Test);
	CPPUNIT_TEST(testFoo);
	CPPUNIT_TEST_SUITE_END();

public:
	void setUp()  // set up
	{
	}

	void tearDown()  // tear down
	{
	}

	void testFoo()
	{
		throw std::runtime_error("not yet implemented");
	}
};

#endif

}  // namespace unit_test
}  // namespace swl

#if defined(__SWL_UNIT_TEST__USE_CPP_UNIT)
//CPPUNIT_TEST_SUITE_REGISTRATION(swl::unit_test::LineSegment3Test);
CPPUNIT_REGISTRY_ADD_TO_DEFAULT("SWL.Math");
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(swl::unit_test::LineSegment3Test, "SWL.Math");
#endif
