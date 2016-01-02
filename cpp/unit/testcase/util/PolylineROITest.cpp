#include "swl/Config.h"
#include "../../UnitTestConfig.h"
#include "swl/base/String.h"
#include "swl/util/RegionOfInterest.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace {
namespace local {

swl::PolylineROI::point_type calculatePoint(const swl::PolylineROI::point_type &lhs, const swl::PolylineROI::point_type &rhs, const swl::PolylineROI::real_type &alpha)
{
	return swl::PolylineROI::point_type((swl::PolylineROI::real_type(1) - alpha) * lhs.x + alpha * rhs.x, (swl::PolylineROI::real_type(1) - alpha) * lhs.y + alpha * rhs.y);
}

}  // namespace local
}  // unnamed namespace

namespace swl {
namespace unit_test {

//-----------------------------------------------------------------------------
// Boost Test

#if defined(__SWL_UNIT_TEST__USE_BOOST_TEST)

namespace {

struct PolylineROITest
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
	void testInclude()
	{
		Fixture fixture;

		const swl::PolylineROI::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f), pt3(10.0f, 30.0f), pt4(21.0f, 25.0f), pt5(28.0f, 3.5f);

		swl::PolylineROI roi(true, swl::PolylineROI::real_type(1), swl::PolylineROI::real_type(1), swl::PolylineROI::color_type(), swl::PolylineROI::color_type());

		roi.addPoint(pt1);
		roi.addPoint(pt2);
		roi.addPoint(pt3);
		roi.addPoint(pt4);
		roi.addPoint(pt5);

		BOOST_CHECK(!roi.include(swl::PolylineROI::point_type(0, 0), swl::PolylineROI::real_type(0.1)));
		BOOST_CHECK(roi.include((pt1 + pt2) / swl::PolylineROI::real_type(2), swl::PolylineROI::real_type(0.1)));
		BOOST_CHECK(roi.include((pt2 + pt3) / swl::PolylineROI::real_type(2), swl::PolylineROI::real_type(0.1)));
		BOOST_CHECK(roi.include((pt3 + pt4) / swl::PolylineROI::real_type(2), swl::PolylineROI::real_type(0.1)));
		BOOST_CHECK(roi.include((pt4 + pt5) / swl::PolylineROI::real_type(2), swl::PolylineROI::real_type(0.1)));
		BOOST_CHECK(!roi.include((pt5 + pt1) / swl::PolylineROI::real_type(2), swl::PolylineROI::real_type(0.1)));
		BOOST_CHECK(!roi.include((pt1 + pt2 + pt3 + pt4 + pt5) / swl::PolylineROI::real_type(5), swl::PolylineROI::real_type(0.1)));

		BOOST_CHECK(roi.include(local::calculatePoint(pt1, pt2, swl::PolylineROI::real_type(0.1)), swl::PolylineROI::real_type(0.01)));
		BOOST_CHECK(roi.include(local::calculatePoint(pt1, pt2, swl::PolylineROI::real_type(0.83)), swl::PolylineROI::real_type(0.01)));
		BOOST_CHECK(!roi.include(local::calculatePoint(pt1, pt2, swl::PolylineROI::real_type(2.1)), swl::PolylineROI::real_type(0.01)));
		BOOST_CHECK(!roi.include(local::calculatePoint(pt1, pt2, swl::PolylineROI::real_type(-15.8)), swl::PolylineROI::real_type(0.01)));
	}
};

struct PolylineROITestSuite: public boost::unit_test_framework::test_suite
{
	PolylineROITestSuite()
	: boost::unit_test_framework::test_suite("SWL.Util.PolylineROI")
	{
		boost::shared_ptr<PolylineROITest> test(new PolylineROITest());

		add(BOOST_CLASS_TEST_CASE(&PolylineROITest::testInclude, test), 0);

		boost::unit_test::framework::master_test_suite().add(this);
	}
} testsuite;

}  // unnamed namespace

//-----------------------------------------------------------------------------
// Google Test

#elif defined(__SWL_UNIT_TEST__USE_GOOGLE_TEST)

class PolylineROITest : public testing::Test
{
protected:
	/*virtual*/ void SetUp()
	{
	}

	/*virtual*/ void TearDown()
	{
	}
};

TEST_F(PolylineROITest, testInclude)
{
	const swl::PolylineROI::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f), pt3(10.0f, 30.0f), pt4(21.0f, 25.0f), pt5(28.0f, 3.5f);

	swl::PolylineROI roi(true, swl::PolylineROI::real_type(1), swl::PolylineROI::real_type(1), swl::PolylineROI::color_type(), swl::PolylineROI::color_type());

	roi.addPoint(pt1);
	roi.addPoint(pt2);
	roi.addPoint(pt3);
	roi.addPoint(pt4);
	roi.addPoint(pt5);

	EXPECT_FALSE(roi.include(swl::PolylineROI::point_type(0, 0), swl::PolylineROI::real_type(0.1)));
	EXPECT_TRUE(roi.include((pt1 + pt2) / swl::PolylineROI::real_type(2), swl::PolylineROI::real_type(0.1)));
	EXPECT_TRUE(roi.include((pt2 + pt3) / swl::PolylineROI::real_type(2), swl::PolylineROI::real_type(0.1)));
	EXPECT_TRUE(roi.include((pt3 + pt4) / swl::PolylineROI::real_type(2), swl::PolylineROI::real_type(0.1)));
	EXPECT_TRUE(roi.include((pt4 + pt5) / swl::PolylineROI::real_type(2), swl::PolylineROI::real_type(0.1)));
	EXPECT_FALSE(roi.include((pt5 + pt1) / swl::PolylineROI::real_type(2), swl::PolylineROI::real_type(0.1)));
	EXPECT_FALSE(roi.include((pt1 + pt2 + pt3 + pt4 + pt5) / swl::PolylineROI::real_type(5), swl::PolylineROI::real_type(0.1)));

	EXPECT_TRUE(roi.include(local::calculatePoint(pt1, pt2, swl::PolylineROI::real_type(0.1)), swl::PolylineROI::real_type(0.01)));
	EXPECT_TRUE(roi.include(local::calculatePoint(pt1, pt2, swl::PolylineROI::real_type(0.83)), swl::PolylineROI::real_type(0.01)));
	EXPECT_FALSE(roi.include(local::calculatePoint(pt1, pt2, swl::PolylineROI::real_type(2.1)), swl::PolylineROI::real_type(0.01)));
	EXPECT_FALSE(roi.include(local::calculatePoint(pt1, pt2, swl::PolylineROI::real_type(-15.8)), swl::PolylineROI::real_type(0.01)));
}

//-----------------------------------------------------------------------------
// CppUnit

#elif defined(__SWL_UNIT_TEST__USE_CPP_UNIT)

struct PolylineROITest : public CppUnit::TestFixture
{
private:
	CPPUNIT_TEST_SUITE(PolylineROITest);
	CPPUNIT_TEST(testInclude);
	CPPUNIT_TEST_SUITE_END();

public:
	void setUp()  // set up
	{
	}

	void tearDown()  // tear down
	{
	}

	void testInclude()
	{
		const swl::PolylineROI::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f), pt3(10.0f, 30.0f), pt4(21.0f, 25.0f), pt5(28.0f, 3.5f);

		swl::PolylineROI roi(true, swl::PolylineROI::real_type(1), swl::PolylineROI::real_type(1), swl::PolylineROI::color_type(), swl::PolylineROI::color_type());

		roi.addPoint(pt1);
		roi.addPoint(pt2);
		roi.addPoint(pt3);
		roi.addPoint(pt4);
		roi.addPoint(pt5);

		CPPUNIT_ASSERT(!roi.include(swl::PolylineROI::point_type(0, 0), swl::PolylineROI::real_type(0.1)));
		CPPUNIT_ASSERT(roi.include((pt1 + pt2) / swl::PolylineROI::real_type(2), swl::PolylineROI::real_type(0.1)));
		CPPUNIT_ASSERT(roi.include((pt2 + pt3) / swl::PolylineROI::real_type(2), swl::PolylineROI::real_type(0.1)));
		CPPUNIT_ASSERT(roi.include((pt3 + pt4) / swl::PolylineROI::real_type(2), swl::PolylineROI::real_type(0.1)));
		CPPUNIT_ASSERT(roi.include((pt4 + pt5) / swl::PolylineROI::real_type(2), swl::PolylineROI::real_type(0.1)));
		CPPUNIT_ASSERT(!roi.include((pt5 + pt1) / swl::PolylineROI::real_type(2), swl::PolylineROI::real_type(0.1)));
		CPPUNIT_ASSERT(!roi.include((pt1 + pt2 + pt3 + pt4 + pt5) / swl::PolylineROI::real_type(5), swl::PolylineROI::real_type(0.1)));

		CPPUNIT_ASSERT(roi.include(local::calculatePoint(pt1, pt2, swl::PolylineROI::real_type(0.1)), swl::PolylineROI::real_type(0.01)));
		CPPUNIT_ASSERT(roi.include(local::calculatePoint(pt1, pt2, swl::PolylineROI::real_type(0.83)), swl::PolylineROI::real_type(0.01)));
		CPPUNIT_ASSERT(!roi.include(local::calculatePoint(pt1, pt2, swl::PolylineROI::real_type(2.1)), swl::PolylineROI::real_type(0.01)));
		CPPUNIT_ASSERT(!roi.include(local::calculatePoint(pt1, pt2, swl::PolylineROI::real_type(-15.8)), swl::PolylineROI::real_type(0.01)));
	}
};

#endif

}  // namespace unit_test
}  // namespace swl

#if defined(__SWL_UNIT_TEST__USE_CPP_UNIT)
//CPPUNIT_TEST_SUITE_REGISTRATION(swl::unit_test::PolylineROITest);
CPPUNIT_REGISTRY_ADD_TO_DEFAULT("SWL.Util");
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(swl::unit_test::PolylineROITest, "SWL.Util");
#endif
