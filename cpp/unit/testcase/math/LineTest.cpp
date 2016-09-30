#include "swl/Config.h"
#include "../../UnitTestConfig.h"
#include "swl/base/String.h"
#include "swl/math/Line.h"
#include <stdexcept>


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

struct Line3Test
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
	void testGetIntersectionPoint()
	{
		Fixture fixture;

		throw std::runtime_error("Not yet implemented");
	}
};

struct LineTestSuite: public boost::unit_test_framework::test_suite
{
	LineTestSuite()
	: boost::unit_test_framework::test_suite("SWL.Math.Line")
	{
		boost::shared_ptr<Line3Test> test(new Line3Test());

		//add(BOOST_CLASS_TEST_CASE(&Line3Test::testGetIntersectionPoint, test), 0);

		boost::unit_test::framework::master_test_suite().add(this);
	}
} testsuite;

}  // unnamed namespace

//-----------------------------------------------------------------------------
// Google Test.

#elif defined(__SWL_UNIT_TEST__USE_GOOGLE_TEST)

class Line3Test : public testing::Test
{
protected:
	/*virtual*/ void SetUp()
	{
	}

	/*virtual*/ void TearDown()
	{
	}
};

//TEST_F(Line3Test, testGetIntersectionPoint)
//{
//	throw std::runtime_error("Not yet implemented");
//}

//-----------------------------------------------------------------------------
// CppUnit.

#elif defined(__SWL_UNIT_TEST__USE_CPP_UNIT)

struct Line3Test: public CppUnit::TestFixture
{
private:
	CPPUNIT_TEST_SUITE(Line3Test);
	//CPPUNIT_TEST(testGetIntersectionPoint);
	CPPUNIT_TEST_SUITE_END();

public:
	void setUp()  // Set up.
	{
	}

	void tearDown()  // Tear down.
	{
	}

	void testGetIntersectionPoint()
	{
		throw std::runtime_error("Not yet implemented");
	}
};

#endif

}  // namespace unit_test
}  // namespace swl

#if defined(__SWL_UNIT_TEST__USE_CPP_UNIT)
//CPPUNIT_TEST_SUITE_REGISTRATION(swl::unit_test::Line3Test);
CPPUNIT_REGISTRY_ADD_TO_DEFAULT("SWL.Math");
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(swl::unit_test::Line3Test, "SWL.Math");
#endif
