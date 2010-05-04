#include "swl/Config.h"
#include "../../UnitTestConfig.h"
#include "swl/base/String.h"
#include "swl/util/RegionOfInterest.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace {

swl::PolygonROI::point_type calculatePoint(const swl::PolygonROI::point_type &lhs, const swl::PolygonROI::point_type &rhs, const swl::PolygonROI::real_type &alpha)
{
	return swl::PolygonROI::point_type((swl::PolygonROI::real_type(1) - alpha) * lhs.x + alpha * rhs.x, (swl::PolygonROI::real_type(1) - alpha) * lhs.y + alpha * rhs.y);
}

}  // unnamed namespace

namespace swl {
namespace unit_test {

//-----------------------------------------------------------------------------
//

#if defined(__SWL_UNIT_TEST__USE_BOOST_UNIT)

namespace {

struct PolygonROITest
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

		const swl::PolygonROI::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f), pt3(10.0f, 30.0f), pt4(21.0f, 25.0f), pt5(28.0f, 3.5f);

		swl::PolygonROI roi(true, swl::PolylineROI::color_type(), swl::PolylineROI::real_type(1));

		roi.addPoint(pt1);
		roi.addPoint(pt2);
		roi.addPoint(pt3);
		roi.addPoint(pt4);
		roi.addPoint(pt5);

		BOOST_CHECK(!roi.include(swl::PolygonROI::point_type(0, 0), swl::PolygonROI::real_type(0.01)));
		BOOST_CHECK(roi.include((pt1 + pt2) / swl::PolygonROI::real_type(2), swl::PolygonROI::real_type(0.01)));
		//BOOST_CHECK(!roi.include((pt2 + pt3) / swl::PolygonROI::real_type(2), swl::PolygonROI::real_type(0.01)));  // caution: not boundary point, but internal point
		BOOST_CHECK(roi.include((pt2 + pt3) / swl::PolygonROI::real_type(2), swl::PolygonROI::real_type(0.01)));
		BOOST_CHECK(roi.include((pt3 + pt4) / swl::PolygonROI::real_type(2), swl::PolygonROI::real_type(0.01)));
		BOOST_CHECK(roi.include((pt4 + pt5) / swl::PolygonROI::real_type(2), swl::PolygonROI::real_type(0.01)));
		//BOOST_CHECK(!roi.include((pt5 + pt1) / swl::PolygonROI::real_type(2), swl::PolygonROI::real_type(0.01)));  // caution: not boundary point, but internal point
		BOOST_CHECK(roi.include((pt5 + pt1) / swl::PolygonROI::real_type(2), swl::PolygonROI::real_type(0.01)));
		BOOST_CHECK(roi.include((pt1 + pt2 + pt3 + pt4 + pt5) / swl::PolylineROI::real_type(5), swl::PolylineROI::real_type(0.01)));

		BOOST_CHECK(roi.include(calculatePoint(pt1, pt2, swl::PolygonROI::real_type(0.1)), swl::PolygonROI::real_type(0.01)));
		BOOST_CHECK(roi.include(calculatePoint(pt1, pt2, swl::PolygonROI::real_type(0.83)), swl::PolygonROI::real_type(0.01)));
		BOOST_CHECK(!roi.include(calculatePoint(pt1, pt2, swl::PolygonROI::real_type(2.1)), swl::PolygonROI::real_type(0.01)));
		BOOST_CHECK(!roi.include(calculatePoint(pt1, pt2, swl::PolygonROI::real_type(-15.8)), swl::PolygonROI::real_type(0.01)));
	}
};

struct PolygonROITestSuite: public boost::unit_test_framework::test_suite
{
	PolygonROITestSuite()
	: boost::unit_test_framework::test_suite("SWL.Util.PolygonROI")
	{
		boost::shared_ptr<PolygonROITest> test(new PolygonROITest());

		add(BOOST_CLASS_TEST_CASE(&PolygonROITest::testInclude, test), 0);

		boost::unit_test::framework::master_test_suite().add(this);
	}
} testsuite;

}  // unnamed namespace

//-----------------------------------------------------------------------------
//

#elif defined(__SWL_UNIT_TEST__USE_CPP_UNIT)

struct PolygonROI: public CppUnit::TestFixture
{
private:
	CPPUNIT_TEST_SUITE(PolygonROI);
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
		const swl::PolygonROI::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f), pt3(10.0f, 30.0f), pt4(21.0f, 25.0f), pt5(28.0f, 3.5f);

		swl::PolygonROI roi(true, swl::PolylineROI::color_type(), swl::PolylineROI::real_type(1));

		roi.addPoint(pt1);
		roi.addPoint(pt2);
		roi.addPoint(pt3);
		roi.addPoint(pt4);
		roi.addPoint(pt5);

		CPPUNIT_ASSERT(!roi.include(swl::PolygonROI::point_type(0, 0), swl::PolygonROI::real_type(0.01)));
		CPPUNIT_ASSERT(roi.include((pt1 + pt2) / swl::PolygonROI::real_type(2), swl::PolygonROI::real_type(0.01)));
		//CPPUNIT_ASSERT(!roi.include((pt2 + pt3) / swl::PolygonROI::real_type(2), swl::PolygonROI::real_type(0.01)));  // caution: not boundary point, but internal point
		CPPUNIT_ASSERT(roi.include((pt2 + pt3) / swl::PolygonROI::real_type(2), swl::PolygonROI::real_type(0.01)));
		CPPUNIT_ASSERT(roi.include((pt3 + pt4) / swl::PolygonROI::real_type(2), swl::PolygonROI::real_type(0.01)));
		CPPUNIT_ASSERT(roi.include((pt4 + pt5) / swl::PolygonROI::real_type(2), swl::PolygonROI::real_type(0.01)));
		//CPPUNIT_ASSERT(!roi.include((pt5 + pt1) / swl::PolygonROI::real_type(2), swl::PolygonROI::real_type(0.01)));  // caution: not boundary point, but internal point
		CPPUNIT_ASSERT(roi.include((pt5 + pt1) / swl::PolygonROI::real_type(2), swl::PolygonROI::real_type(0.01)));
		CPPUNIT_ASSERT(roi.include((pt1 + pt2 + pt3 + pt4 + pt5) / swl::PolylineROI::real_type(5), swl::PolylineROI::real_type(0.01)));

		CPPUNIT_ASSERT(roi.include(calculatePoint(pt1, pt2, swl::PolygonROI::real_type(0.1)), swl::PolygonROI::real_type(0.01)));
		CPPUNIT_ASSERT(roi.include(calculatePoint(pt1, pt2, swl::PolygonROI::real_type(0.83)), swl::PolygonROI::real_type(0.01)));
		CPPUNIT_ASSERT(!roi.include(calculatePoint(pt1, pt2, swl::PolygonROI::real_type(2.1)), swl::PolygonROI::real_type(0.01)));
		CPPUNIT_ASSERT(!roi.include(calculatePoint(pt1, pt2, swl::PolygonROI::real_type(-15.8)), swl::PolygonROI::real_type(0.01)));
	}
};

#endif

}  // namespace unit_test
}  // namespace swl

#if defined(__SWL_UNIT_TEST__USE_CPP_UNIT)
//CPPUNIT_TEST_SUITE_REGISTRATION(swl::unit_test::PolygonROI);
CPPUNIT_REGISTRY_ADD_TO_DEFAULT("SWL.Util");
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(swl::unit_test::PolygonROI, "SWL.Util");
#endif
