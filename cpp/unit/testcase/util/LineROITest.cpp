#include "swl/Config.h"
#include "../../UnitTestConfig.h"
#include "swl/base/String.h"
#include "swl/util/RegionOfInterest.h"
#include <cmath>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace {
namespace local {

bool comparePoints(const swl::LineROI::point_type &lhs, const swl::LineROI::point_type &rhs)
{
	const swl::LineROI::real_type eps = swl::LineROI::real_type(1.0e-15);
	return std::fabs(lhs.x - rhs.x) <= eps && std::fabs(lhs.y - rhs.y) <= eps;
}

swl::LineROI::point_type calculatePoint(const swl::LineROI::point_type &lhs, const swl::LineROI::point_type &rhs, const swl::LineROI::real_type &alpha)
{
	return swl::LineROI::point_type((swl::LineROI::real_type(1) - alpha) * lhs.x + alpha * rhs.x, (swl::LineROI::real_type(1) - alpha) * lhs.y + alpha * rhs.y);
}

}  // namespace local
}  // unnamed namespace

namespace swl {
namespace unit_test {

//-----------------------------------------------------------------------------
// Boost Test

#if defined(__SWL_UNIT_TEST__USE_BOOST_TEST)

namespace {

struct LineROITest
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
	void testMoveVertex()
	{
		Fixture fixture;

		const swl::LineROI::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f);
		const swl::LineROI::point_type delta(3.0f, -7.0f);

		{
			swl::LineROI roi(pt1, pt2, true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type());
			BOOST_CHECK(!roi.moveVertex(swl::LineROI::point_type(-21.0f, 10.0f), delta, 0.1f));
			BOOST_CHECK(local::comparePoints(roi.point1(), pt1));
			BOOST_CHECK(local::comparePoints(roi.point2(), pt2));

			BOOST_CHECK(roi.moveVertex(swl::LineROI::point_type(-21.0f, 10.0f), delta, 2.0f));
			BOOST_CHECK(local::comparePoints(roi.point1(), pt1 + delta));
			BOOST_CHECK(!local::comparePoints(roi.point2(), pt2 + delta));
			BOOST_CHECK(local::comparePoints(roi.point2(), pt2));
		}

		{
			swl::LineROI roi(pt1, pt2, true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type());
			BOOST_CHECK(!roi.moveVertex(swl::LineROI::point_type(39.0f, 26.0f), delta, 0.1f));
			BOOST_CHECK(local::comparePoints(roi.point1(), pt1));
			BOOST_CHECK(local::comparePoints(roi.point2(), pt2));

			BOOST_CHECK(roi.moveVertex(swl::LineROI::point_type(39.0f, 26.0f), delta, 2.0f));
			BOOST_CHECK(!local::comparePoints(roi.point1(), pt1 + delta));
			BOOST_CHECK(local::comparePoints(roi.point1(), pt1));
			BOOST_CHECK(local::comparePoints(roi.point2(), pt2 + delta));
		}
	}

	void testMoveVertexWithLimit()
	{
		Fixture fixture;

		const swl::LineROI::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f);
		const swl::LineROI::point_type delta(5.0f, 10.0f);
		const swl::LineROI::point_type bigDelta(100.0f, -100.0f);
		const swl::LineROI::region_type limitRegion(swl::LineROI::point_type(-5.0f, -5.0f), swl::LineROI::point_type(50.0f, 50.0f));

		{
			swl::LineROI roi(pt1, pt2, true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type());
			BOOST_CHECK(!roi.moveVertex(swl::LineROI::point_type(-21.0f, 10.0f), delta, limitRegion, 0.1f));
			BOOST_CHECK(local::comparePoints(roi.point1(), pt1));
			BOOST_CHECK(local::comparePoints(roi.point2(), pt2));

			BOOST_CHECK(roi.moveVertex(swl::LineROI::point_type(-21.0f, 10.0f), delta, limitRegion, 2.0f));
			BOOST_CHECK(local::comparePoints(roi.point1(), pt1 + delta));
			BOOST_CHECK(local::comparePoints(roi.point2(), pt2));
		}

		{
			swl::LineROI roi(pt1, pt2, true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type());
			BOOST_CHECK(!roi.moveVertex(swl::LineROI::point_type(39.0f, 26.0f), delta, limitRegion, 0.1f));
			BOOST_CHECK(local::comparePoints(roi.point1(), pt1));
			BOOST_CHECK(local::comparePoints(roi.point2(), pt2));

			BOOST_CHECK(roi.moveVertex(swl::LineROI::point_type(39.0f, 26.0f), delta, limitRegion, 2.0f));
			BOOST_CHECK(local::comparePoints(roi.point1(), pt1));
			BOOST_CHECK(local::comparePoints(roi.point2(), pt2 + delta));
		}

		{
			swl::LineROI roi(pt1, pt2, true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type());
			BOOST_CHECK(roi.moveVertex(swl::LineROI::point_type(-21.0f, 10.0f), bigDelta, limitRegion, 2.0f));
			BOOST_CHECK(local::comparePoints(roi.point1(), swl::LineROI::point_type(limitRegion.right, limitRegion.bottom)));
			BOOST_CHECK(!local::comparePoints(roi.point2(), swl::LineROI::point_type(limitRegion.right, limitRegion.bottom)));
			BOOST_CHECK(local::comparePoints(roi.point2(), pt2));
		}

		{
			swl::LineROI roi(pt1, pt2, true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type());
			BOOST_CHECK(roi.moveVertex(swl::LineROI::point_type(39.0f, 26.0f), bigDelta, limitRegion, 2.0f));
			BOOST_CHECK(!local::comparePoints(roi.point1(), swl::LineROI::point_type(limitRegion.right, limitRegion.bottom)));
			BOOST_CHECK(local::comparePoints(roi.point1(), pt1));
			BOOST_CHECK(local::comparePoints(roi.point2(), swl::LineROI::point_type(limitRegion.right, limitRegion.bottom)));
		}
	}

	void testMoveRegion()
	{
		Fixture fixture;

		const swl::LineROI::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f);
		const swl::LineROI::point_type delta(3.0f, -7.0f);

		swl::LineROI roi(pt1, pt2, true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type());
		roi.moveRegion(delta);
		BOOST_CHECK(local::comparePoints(roi.point1(), pt1 + delta));
		BOOST_CHECK(local::comparePoints(roi.point2(), pt2 + delta));
	}

	void testMoveRegionWithLimit()
	{
		Fixture fixture;

		const swl::LineROI::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f);
		const swl::LineROI::region_type limitRegion(swl::LineROI::point_type(-5.0f, -5.0f), swl::LineROI::point_type(50.0f, 50.0f));

		{
			const swl::LineROI::point_type delta(5.0f, 10.0f);

			swl::LineROI roi(pt1, pt2, true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type());
			roi.moveRegion(delta, limitRegion);
			BOOST_CHECK(local::comparePoints(roi.point1(), pt1 + delta));
			BOOST_CHECK(local::comparePoints(roi.point2(), pt2 + delta));
		}

		{
			const swl::LineROI::point_type bigDelta(100.0f, -100.0f);
			const swl::LineROI::real_type dx = 10.0f, dy = -15.0f;  // actual displacement

			swl::LineROI roi(pt1, pt2, true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type());
			roi.moveRegion(bigDelta, limitRegion);
			BOOST_CHECK(local::comparePoints(roi.point1(), pt1 + swl::LineROI::point_type(dx, dy)));  // caution: not (-5, -5), but (-10, -5)
			BOOST_CHECK(local::comparePoints(roi.point2(), pt2 + swl::LineROI::point_type(dx, dy)));
		}

		{
			const swl::LineROI::point_type delta(-5.0f, 100.0f);
			const swl::LineROI::real_type dx = -5.0f, dy = 25.0f;  // computed displacement
			const swl::LineROI::real_type dx2 = 0.0f;  // actual displacement: don't move along the negative x-axis.

			swl::LineROI roi(pt1, pt2, true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type());
			roi.moveRegion(delta, limitRegion);
			BOOST_CHECK(!local::comparePoints(roi.point1(), pt1 + swl::LineROI::point_type(dx, dy)));  // caution: not (-25, 35), but (-20, 35)  <==  don't move along the negative x-axis because x-value of pt1 is beyond a limit region.
			BOOST_CHECK(local::comparePoints(roi.point1(), pt1 + swl::LineROI::point_type(dx2, dy)));
			BOOST_CHECK(!local::comparePoints(roi.point2(), pt2 + swl::LineROI::point_type(dx, dy)));
			BOOST_CHECK(local::comparePoints(roi.point2(), pt2 + swl::LineROI::point_type(dx2, dy)));
		}
	}

	void testIsVertex()
	{
		Fixture fixture;

		const swl::LineROI::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f);
		const swl::LineROI roi(pt1, pt2, true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type());

		BOOST_CHECK(roi.isVertex(swl::LineROI::point_type(-20.1f, 10.0f), swl::LineROI::real_type(0.5)));
		BOOST_CHECK(!roi.isVertex(swl::LineROI::point_type(-20.1f, 10.0f), swl::LineROI::real_type(0.05)));
		BOOST_CHECK(roi.isVertex(swl::LineROI::point_type(41.0f, 23.5f), swl::LineROI::real_type(3)));
		BOOST_CHECK(!roi.isVertex(swl::LineROI::point_type(40.1f, 25.3f), swl::LineROI::real_type(0.2)));

		BOOST_CHECK(!roi.isVertex((pt1 + pt2) / swl::RectangleROI::real_type(2), swl::LineROI::real_type(1)));
	}

	void testInclude()
	{
		Fixture fixture;

		const swl::LineROI::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f);
		const swl::LineROI roi(pt1, pt2, true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type());

		BOOST_CHECK(!roi.include(swl::LineROI::point_type(0, 0), swl::LineROI::real_type(0.01)));
		BOOST_CHECK(roi.include((pt1 + pt2) / swl::LineROI::real_type(2), swl::LineROI::real_type(0.01)));

		BOOST_CHECK(roi.include(local::calculatePoint(pt1, pt2, swl::LineROI::real_type(0.1)), swl::LineROI::real_type(0.01)));
		BOOST_CHECK(roi.include(local::calculatePoint(pt1, pt2, swl::LineROI::real_type(0.83)), swl::LineROI::real_type(0.01)));
		BOOST_CHECK(!roi.include(local::calculatePoint(pt1, pt2, swl::LineROI::real_type(2.1)), swl::LineROI::real_type(0.01)));
		BOOST_CHECK(!roi.include(local::calculatePoint(pt1, pt2, swl::LineROI::real_type(-15.8)), swl::LineROI::real_type(0.01)));
	}
};

struct LineROITestSuite: public boost::unit_test_framework::test_suite
{
	LineROITestSuite()
	: boost::unit_test_framework::test_suite("SWL.Util.LineROI")
	{
		boost::shared_ptr<LineROITest> test(new LineROITest());

		add(BOOST_CLASS_TEST_CASE(&LineROITest::testMoveVertex, test), 0);
		add(BOOST_CLASS_TEST_CASE(&LineROITest::testMoveVertexWithLimit, test), 0);
		add(BOOST_CLASS_TEST_CASE(&LineROITest::testMoveRegion, test), 0);
		add(BOOST_CLASS_TEST_CASE(&LineROITest::testMoveRegionWithLimit, test), 0);
		add(BOOST_CLASS_TEST_CASE(&LineROITest::testIsVertex, test), 0);
		add(BOOST_CLASS_TEST_CASE(&LineROITest::testInclude, test), 0);

		boost::unit_test::framework::master_test_suite().add(this);
	}
} testsuite;

}  // unnamed namespace

//-----------------------------------------------------------------------------
// Google Test

#elif defined(__SWL_UNIT_TEST__USE_GOOGLE_TEST)

class LineROITest : public testing::Test
{
protected:
	/*virtual*/ void SetUp()
	{
	}

	/*virtual*/ void TearDown()
	{
	}
};

TEST_F(LineROITest, testMoveVertex)
{
	const swl::LineROI::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f);
	const swl::LineROI::point_type delta(3.0f, -7.0f);

	{
		swl::LineROI roi(pt1, pt2, true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type());
		EXPECT_FALSE(roi.moveVertex(swl::LineROI::point_type(-21.0f, 10.0f), delta, 0.1f));
		EXPECT_TRUE(local::comparePoints(roi.point1(), pt1));
		EXPECT_TRUE(local::comparePoints(roi.point2(), pt2));

		EXPECT_TRUE(roi.moveVertex(swl::LineROI::point_type(-21.0f, 10.0f), delta, 2.0f));
		EXPECT_TRUE(local::comparePoints(roi.point1(), pt1 + delta));
		EXPECT_FALSE(local::comparePoints(roi.point2(), pt2 + delta));
		EXPECT_TRUE(local::comparePoints(roi.point2(), pt2));
	}

	{
		swl::LineROI roi(pt1, pt2, true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type());
		EXPECT_FALSE(roi.moveVertex(swl::LineROI::point_type(39.0f, 26.0f), delta, 0.1f));
		EXPECT_TRUE(local::comparePoints(roi.point1(), pt1));
		EXPECT_TRUE(local::comparePoints(roi.point2(), pt2));

		EXPECT_TRUE(roi.moveVertex(swl::LineROI::point_type(39.0f, 26.0f), delta, 2.0f));
		EXPECT_FALSE(local::comparePoints(roi.point1(), pt1 + delta));
		EXPECT_TRUE(local::comparePoints(roi.point1(), pt1));
		EXPECT_TRUE(local::comparePoints(roi.point2(), pt2 + delta));
	}
}

TEST_F(LineROITest, testMoveVertexWithLimit)
{
	const swl::LineROI::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f);
	const swl::LineROI::point_type delta(5.0f, 10.0f);
	const swl::LineROI::point_type bigDelta(100.0f, -100.0f);
	const swl::LineROI::region_type limitRegion(swl::LineROI::point_type(-5.0f, -5.0f), swl::LineROI::point_type(50.0f, 50.0f));

	{
		swl::LineROI roi(pt1, pt2, true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type());
		EXPECT_FALSE(roi.moveVertex(swl::LineROI::point_type(-21.0f, 10.0f), delta, limitRegion, 0.1f));
		EXPECT_TRUE(local::comparePoints(roi.point1(), pt1));
		EXPECT_TRUE(local::comparePoints(roi.point2(), pt2));

		EXPECT_TRUE(roi.moveVertex(swl::LineROI::point_type(-21.0f, 10.0f), delta, limitRegion, 2.0f));
		EXPECT_TRUE(local::comparePoints(roi.point1(), pt1 + delta));
		EXPECT_TRUE(local::comparePoints(roi.point2(), pt2));
	}

	{
		swl::LineROI roi(pt1, pt2, true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type());
		EXPECT_FALSE(roi.moveVertex(swl::LineROI::point_type(39.0f, 26.0f), delta, limitRegion, 0.1f));
		EXPECT_TRUE(local::comparePoints(roi.point1(), pt1));
		EXPECT_TRUE(local::comparePoints(roi.point2(), pt2));

		EXPECT_TRUE(roi.moveVertex(swl::LineROI::point_type(39.0f, 26.0f), delta, limitRegion, 2.0f));
		EXPECT_TRUE(local::comparePoints(roi.point1(), pt1));
		EXPECT_TRUE(local::comparePoints(roi.point2(), pt2 + delta));
	}

	{
		swl::LineROI roi(pt1, pt2, true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type());
		EXPECT_TRUE(roi.moveVertex(swl::LineROI::point_type(-21.0f, 10.0f), bigDelta, limitRegion, 2.0f));
		EXPECT_TRUE(local::comparePoints(roi.point1(), swl::LineROI::point_type(limitRegion.right, limitRegion.bottom)));
		EXPECT_FALSE(local::comparePoints(roi.point2(), swl::LineROI::point_type(limitRegion.right, limitRegion.bottom)));
		EXPECT_TRUE(local::comparePoints(roi.point2(), pt2));
	}

	{
		swl::LineROI roi(pt1, pt2, true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type());
		EXPECT_TRUE(roi.moveVertex(swl::LineROI::point_type(39.0f, 26.0f), bigDelta, limitRegion, 2.0f));
		EXPECT_FALSE(local::comparePoints(roi.point1(), swl::LineROI::point_type(limitRegion.right, limitRegion.bottom)));
		EXPECT_TRUE(local::comparePoints(roi.point1(), pt1));
		EXPECT_TRUE(local::comparePoints(roi.point2(), swl::LineROI::point_type(limitRegion.right, limitRegion.bottom)));
	}
}

TEST_F(LineROITest, testMoveRegion)
{
	const swl::LineROI::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f);
	const swl::LineROI::point_type delta(3.0f, -7.0f);

	swl::LineROI roi(pt1, pt2, true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type());
	roi.moveRegion(delta);
	EXPECT_TRUE(local::comparePoints(roi.point1(), pt1 + delta));
	EXPECT_TRUE(local::comparePoints(roi.point2(), pt2 + delta));
}

TEST_F(LineROITest, testMoveRegionWithLimit)
{
	const swl::LineROI::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f);
	const swl::LineROI::region_type limitRegion(swl::LineROI::point_type(-5.0f, -5.0f), swl::LineROI::point_type(50.0f, 50.0f));

	{
		const swl::LineROI::point_type delta(5.0f, 10.0f);

		swl::LineROI roi(pt1, pt2, true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type());
		roi.moveRegion(delta, limitRegion);
		EXPECT_TRUE(local::comparePoints(roi.point1(), pt1 + delta));
		EXPECT_TRUE(local::comparePoints(roi.point2(), pt2 + delta));
	}

	{
		const swl::LineROI::point_type bigDelta(100.0f, -100.0f);
		const swl::LineROI::real_type dx = 10.0f, dy = -15.0f;  // actual displacement

		swl::LineROI roi(pt1, pt2, true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type());
		roi.moveRegion(bigDelta, limitRegion);
		EXPECT_TRUE(local::comparePoints(roi.point1(), pt1 + swl::LineROI::point_type(dx, dy)));  // caution: not (-5, -5), but (-10, -5)
		EXPECT_TRUE(local::comparePoints(roi.point2(), pt2 + swl::LineROI::point_type(dx, dy)));
	}

	{
		const swl::LineROI::point_type delta(-5.0f, 100.0f);
		const swl::LineROI::real_type dx = -5.0f, dy = 25.0f;  // computed displacement
		const swl::LineROI::real_type dx2 = 0.0f;  // actual displacement: don't move along the negative x-axis.

		swl::LineROI roi(pt1, pt2, true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type());
		roi.moveRegion(delta, limitRegion);
		EXPECT_FALSE(local::comparePoints(roi.point1(), pt1 + swl::LineROI::point_type(dx, dy)));  // caution: not (-25, 35), but (-20, 35)  <==  don't move along the negative x-axis because x-value of pt1 is beyond a limit region.
		EXPECT_TRUE(local::comparePoints(roi.point1(), pt1 + swl::LineROI::point_type(dx2, dy)));
		EXPECT_FALSE(local::comparePoints(roi.point2(), pt2 + swl::LineROI::point_type(dx, dy)));
		EXPECT_TRUE(local::comparePoints(roi.point2(), pt2 + swl::LineROI::point_type(dx2, dy)));
	}
}

TEST_F(LineROITest, testIsVertex)
{
	const swl::LineROI::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f);
	const swl::LineROI roi(pt1, pt2, true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type());

	EXPECT_TRUE(roi.isVertex(swl::LineROI::point_type(-20.1f, 10.0f), swl::LineROI::real_type(0.5)));
	EXPECT_FALSE(roi.isVertex(swl::LineROI::point_type(-20.1f, 10.0f), swl::LineROI::real_type(0.05)));
	EXPECT_TRUE(roi.isVertex(swl::LineROI::point_type(41.0f, 23.5f), swl::LineROI::real_type(3)));
	EXPECT_FALSE(roi.isVertex(swl::LineROI::point_type(40.1f, 25.3f), swl::LineROI::real_type(0.2)));

	EXPECT_FALSE(roi.isVertex((pt1 + pt2) / swl::RectangleROI::real_type(2), swl::LineROI::real_type(1)));
}

TEST_F(LineROITest, testInclude)
{
	const swl::LineROI::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f);
	const swl::LineROI roi(pt1, pt2, true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type());

	EXPECT_FALSE(roi.include(swl::LineROI::point_type(0, 0), swl::LineROI::real_type(0.01)));
	EXPECT_TRUE(roi.include((pt1 + pt2) / swl::LineROI::real_type(2), swl::LineROI::real_type(0.01)));

	EXPECT_TRUE(roi.include(local::calculatePoint(pt1, pt2, swl::LineROI::real_type(0.1)), swl::LineROI::real_type(0.01)));
	EXPECT_TRUE(roi.include(local::calculatePoint(pt1, pt2, swl::LineROI::real_type(0.83)), swl::LineROI::real_type(0.01)));
	EXPECT_FALSE(roi.include(local::calculatePoint(pt1, pt2, swl::LineROI::real_type(2.1)), swl::LineROI::real_type(0.01)));
	EXPECT_FALSE(roi.include(local::calculatePoint(pt1, pt2, swl::LineROI::real_type(-15.8)), swl::LineROI::real_type(0.01)));
}

//-----------------------------------------------------------------------------
// CppUnit

#elif defined(__SWL_UNIT_TEST__USE_CPP_UNIT)

struct LineROITest: public CppUnit::TestFixture
{
private:
	CPPUNIT_TEST_SUITE(LineROITest);
	CPPUNIT_TEST(testMoveVertex);
	CPPUNIT_TEST(testMoveVertexWithLimit);
	CPPUNIT_TEST(testMoveRegion);
	CPPUNIT_TEST(testMoveRegionWithLimit);
	CPPUNIT_TEST(testIsVertex);
	CPPUNIT_TEST(testInclude);
	CPPUNIT_TEST_SUITE_END();

public:
	void setUp()  // set up
	{
	}

	void tearDown()  // tear down
	{
	}

	void testMoveVertex()
	{
		const swl::LineROI::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f);
		const swl::LineROI::point_type delta(3.0f, -7.0f);

		{
			swl::LineROI roi(pt1, pt2, true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type());
			CPPUNIT_ASSERT(!roi.moveVertex(swl::LineROI::point_type(-21.0f, 10.0f), delta, 0.1f));
			CPPUNIT_ASSERT(local::comparePoints(roi.point1(), pt1));
			CPPUNIT_ASSERT(local::comparePoints(roi.point2(), pt2));

			CPPUNIT_ASSERT(roi.moveVertex(swl::LineROI::point_type(-21.0f, 10.0f), delta, 2.0f));
			CPPUNIT_ASSERT(local::comparePoints(roi.point1(), pt1 + delta));
			CPPUNIT_ASSERT(!local::comparePoints(roi.point2(), pt2 + delta));
			CPPUNIT_ASSERT(local::comparePoints(roi.point2(), pt2));
		}

		{
			swl::LineROI roi(pt1, pt2, true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type());
			CPPUNIT_ASSERT(!roi.moveVertex(swl::LineROI::point_type(39.0f, 26.0f), delta, 0.1f));
			CPPUNIT_ASSERT(local::comparePoints(roi.point1(), pt1));
			CPPUNIT_ASSERT(local::comparePoints(roi.point1(), pt1));

			CPPUNIT_ASSERT(roi.moveVertex(swl::LineROI::point_type(39.0f, 26.0f), delta, 2.0f));
			CPPUNIT_ASSERT(!local::comparePoints(roi.point1(), pt1 + delta));
			CPPUNIT_ASSERT(local::comparePoints(roi.point1(), pt1));
			CPPUNIT_ASSERT(local::comparePoints(roi.point2(), pt2 + delta));
		}
	}

	void testMoveVertexWithLimit()
	{
		const swl::LineROI::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f);
		const swl::LineROI::point_type delta(5.0f, 10.0f);
		const swl::LineROI::point_type bigDelta(100.0f, -100.0f);
		const swl::LineROI::region_type limitRegion(swl::LineROI::point_type(-5.0f, -5.0f), swl::LineROI::point_type(50.0f, 50.0f));

		{
			swl::LineROI roi(pt1, pt2, true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type());
			CPPUNIT_ASSERT(!roi.moveVertex(swl::LineROI::point_type(-21.0f, 10.0f), delta, limitRegion, 0.1f));
			CPPUNIT_ASSERT(local::comparePoints(roi.point1(), pt1));
			CPPUNIT_ASSERT(local::comparePoints(roi.point2(), pt2));

			CPPUNIT_ASSERT(roi.moveVertex(swl::LineROI::point_type(-21.0f, 10.0f), delta, limitRegion, 2.0f));
			CPPUNIT_ASSERT(local::comparePoints(roi.point1(), pt1 + delta));
			CPPUNIT_ASSERT(local::comparePoints(roi.point2(), pt2));
		}

		{
			swl::LineROI roi(pt1, pt2, true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type());
			CPPUNIT_ASSERT(!roi.moveVertex(swl::LineROI::point_type(39.0f, 26.0f), delta, limitRegion, 0.1f));
			CPPUNIT_ASSERT(local::comparePoints(roi.point1(), pt1));
			CPPUNIT_ASSERT(local::comparePoints(roi.point2(), pt2));

			CPPUNIT_ASSERT(roi.moveVertex(swl::LineROI::point_type(39.0f, 26.0f), delta, limitRegion, 2.0f));
			CPPUNIT_ASSERT(local::comparePoints(roi.point1(), pt1));
			CPPUNIT_ASSERT(local::comparePoints(roi.point2(), pt2 + delta));
		}

		{
			swl::LineROI roi(pt1, pt2, true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type());
			CPPUNIT_ASSERT(roi.moveVertex(swl::LineROI::point_type(-21.0f, 10.0f), bigDelta, limitRegion, 2.0f));
			CPPUNIT_ASSERT(local::comparePoints(roi.point1(), swl::LineROI::point_type(limitRegion.right, limitRegion.bottom)));
			CPPUNIT_ASSERT(!local::comparePoints(roi.point2(), swl::LineROI::point_type(limitRegion.right, limitRegion.bottom)));
			CPPUNIT_ASSERT(local::comparePoints(roi.point2(), pt2));
		}

		{
			swl::LineROI roi(pt1, pt2, true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type());
			CPPUNIT_ASSERT(roi.moveVertex(swl::LineROI::point_type(39.0f, 26.0f), bigDelta, limitRegion, 2.0f));
			CPPUNIT_ASSERT(!local::comparePoints(roi.point1(), swl::LineROI::point_type(limitRegion.right, limitRegion.bottom)));
			CPPUNIT_ASSERT(local::comparePoints(roi.point1(), pt1));
			CPPUNIT_ASSERT(local::comparePoints(roi.point2(), swl::LineROI::point_type(limitRegion.right, limitRegion.bottom)));
		}
	}

	void testMoveRegion()
	{
		const swl::LineROI::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f);
		const swl::LineROI::point_type delta(3.0f, -7.0f);

		swl::LineROI roi(pt1, pt2, true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type());
		roi.moveRegion(delta);
		CPPUNIT_ASSERT(local::comparePoints(roi.point1(), pt1 + delta));
		CPPUNIT_ASSERT(local::comparePoints(roi.point2(), pt2 + delta));
	}

	void testMoveRegionWithLimit()
	{
		const swl::LineROI::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f);
		const swl::LineROI::region_type limitRegion(swl::LineROI::point_type(-5.0f, -5.0f), swl::LineROI::point_type(50.0f, 50.0f));

		{
			const swl::LineROI::point_type delta(5.0f, 10.0f);

			swl::LineROI roi(pt1, pt2, true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type());
			roi.moveRegion(delta, limitRegion);
			CPPUNIT_ASSERT(local::comparePoints(roi.point1(), pt1 + delta));
			CPPUNIT_ASSERT(local::comparePoints(roi.point2(), pt2 + delta));
		}

		{
			const swl::LineROI::point_type bigDelta(100.0f, -100.0f);
			const swl::LineROI::real_type dx = 10.0f, dy = -15.0f;  // actual displacement

			swl::LineROI roi(pt1, pt2, true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type());
			roi.moveRegion(bigDelta, limitRegion);
			CPPUNIT_ASSERT(local::comparePoints(roi.point1(), pt1 + swl::LineROI::point_type(dx, dy)));  // caution: not (-5, -5), but (-10, -5)
			CPPUNIT_ASSERT(local::comparePoints(roi.point2(), pt2 + swl::LineROI::point_type(dx, dy)));
		}

		{
			const swl::LineROI::point_type delta(-5.0f, 100.0f);
			const swl::LineROI::real_type dx = -5.0f, dy = 25.0f;  // computed displacement
			const swl::LineROI::real_type dx2 = 0.0f;  // actual displacement: don't move along the negative x-axis.

			swl::LineROI roi(pt1, pt2, true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type());
			roi.moveRegion(delta, limitRegion);
			CPPUNIT_ASSERT(!local::comparePoints(roi.point1(), pt1 + swl::LineROI::point_type(dx, dy)));  // caution: not (-25, 35), but (-20, 35)  <==  don't move along the negative x-axis because x-value of pt1 is beyond a limit region.
			CPPUNIT_ASSERT(local::comparePoints(roi.point1(), pt1 + swl::LineROI::point_type(dx2, dy)));
			CPPUNIT_ASSERT(!local::comparePoints(roi.point2(), pt2 + swl::LineROI::point_type(dx, dy)));
			CPPUNIT_ASSERT(local::comparePoints(roi.point2(), pt2 + swl::LineROI::point_type(dx2, dy)));
		}
	}

	void testIsVertex()
	{
		const swl::LineROI::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f);
		const swl::LineROI roi(pt1, pt2, true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type());

		CPPUNIT_ASSERT(roi.isVertex(swl::LineROI::point_type(-20.1f, 10.0f), swl::LineROI::real_type(0.5)));
		CPPUNIT_ASSERT(!roi.isVertex(swl::LineROI::point_type(-20.1f, 10.0f), swl::LineROI::real_type(0.05)));
		CPPUNIT_ASSERT(roi.isVertex(swl::LineROI::point_type(41.0f, 23.5f), swl::LineROI::real_type(3)));
		CPPUNIT_ASSERT(!roi.isVertex(swl::LineROI::point_type(40.1f, 25.3f), swl::LineROI::real_type(0.2)));

		CPPUNIT_ASSERT(!roi.isVertex((pt1 + pt2) / swl::RectangleROI::real_type(2), swl::LineROI::real_type(1)));
	}

	void testInclude()
	{
		const swl::LineROI::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f);
		const swl::LineROI roi(pt1, pt2, true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type());

		CPPUNIT_ASSERT(!roi.include(swl::LineROI::point_type(0, 0), swl::LineROI::real_type(0.01)));
		CPPUNIT_ASSERT(roi.include((pt1 + pt2) / swl::LineROI::real_type(2), swl::LineROI::real_type(0.01)));

		CPPUNIT_ASSERT(roi.include(local::calculatePoint(pt1, pt2, swl::LineROI::real_type(0.1)), swl::LineROI::real_type(0.01)));
		CPPUNIT_ASSERT(roi.include(local::calculatePoint(pt1, pt2, swl::LineROI::real_type(0.83)), swl::LineROI::real_type(0.01)));
		CPPUNIT_ASSERT(!roi.include(local::calculatePoint(pt1, pt2, swl::LineROI::real_type(2.1)), swl::LineROI::real_type(0.01)));
		CPPUNIT_ASSERT(!roi.include(local::calculatePoint(pt1, pt2, swl::LineROI::real_type(-15.8)), swl::LineROI::real_type(0.01)));
	}
};

#endif

}  // namespace unit_test
}  // namespace swl

#if defined(__SWL_UNIT_TEST__USE_CPP_UNIT)
//CPPUNIT_TEST_SUITE_REGISTRATION(swl::unit_test::LineROITest);
CPPUNIT_REGISTRY_ADD_TO_DEFAULT("SWL.Util");
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(swl::unit_test::LineROITest, "SWL.Util");
#endif
