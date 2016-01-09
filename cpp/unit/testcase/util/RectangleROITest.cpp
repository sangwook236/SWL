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

bool comparePoints(const swl::RectangleROI::point_type &lhs, const swl::RectangleROI::point_type &rhs, const swl::RectangleROI::real_type &eps = swl::RectangleROI::real_type(1.0e-15))
{
	return std::abs(lhs.x - rhs.x) <= eps && std::abs(lhs.y - rhs.y) <= eps;
}

swl::RectangleROI::point_type calculatePoint(const swl::RectangleROI::point_type &lhs, const swl::RectangleROI::point_type &rhs, const swl::RectangleROI::real_type &alpha)
{
	return swl::RectangleROI::point_type((swl::RectangleROI::real_type(1) - alpha) * lhs.x + alpha * rhs.x, (swl::RectangleROI::real_type(1) - alpha) * lhs.y + alpha * rhs.y);
}

}  // namespace local
}  // unnamed namespace

namespace swl {
namespace unit_test {

//-----------------------------------------------------------------------------
// Boost Test

#if defined(__SWL_UNIT_TEST__USE_BOOST_TEST)

namespace {

struct RectangleROITest
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

		const swl::RectangleROI::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f);
		const swl::RectangleROI::point_type delta(3.0f, -7.0f);

		{
			swl::RectangleROI roi(pt1, pt2, true, swl::RectangleROI::real_type(1), swl::RectangleROI::real_type(1), swl::RectangleROI::color_type(), swl::RectangleROI::color_type());
			BOOST_CHECK(!roi.moveVertex(swl::RectangleROI::point_type(-21.0f, 10.0f), delta, 0.1f));
			BOOST_CHECK(local::comparePoints(roi.point1(), pt1));
			BOOST_CHECK(local::comparePoints(roi.point2(), pt2));

			BOOST_CHECK(roi.moveVertex(swl::RectangleROI::point_type(-21.0f, 10.0f), delta, 2.0f));
			BOOST_CHECK(local::comparePoints(roi.point1(), pt1 + delta));
			BOOST_CHECK(!local::comparePoints(roi.point2(), pt2 + delta));
			BOOST_CHECK(local::comparePoints(roi.point2(), pt2));
		}

		{
			swl::RectangleROI roi(pt1, pt2, true, swl::RectangleROI::real_type(1), swl::RectangleROI::real_type(1), swl::RectangleROI::color_type(), swl::RectangleROI::color_type());
			BOOST_CHECK(!roi.moveVertex(swl::RectangleROI::point_type(39.0f, 26.0f), delta, 0.1f));
			BOOST_CHECK(local::comparePoints(roi.point1(), pt1));
			BOOST_CHECK(local::comparePoints(roi.point2(), pt2));

			BOOST_CHECK(roi.moveVertex(swl::RectangleROI::point_type(39.0f, 26.0f), delta, 2.0f));
			BOOST_CHECK(!local::comparePoints(roi.point1(), pt1 + delta));
			BOOST_CHECK(local::comparePoints(roi.point1(), pt1));
			BOOST_CHECK(local::comparePoints(roi.point2(), pt2 + delta));
		}
	}

	void testMoveVertexWithLimit()
	{
		Fixture fixture;

		const swl::RectangleROI::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f);
		const swl::RectangleROI::point_type delta(5.0f, 10.0f);
		const swl::RectangleROI::point_type bigDelta(100.0f, -100.0f);
		const swl::RectangleROI::region_type limitRegion(swl::RectangleROI::point_type(-5.0f, -5.0f), swl::RectangleROI::point_type(50.0f, 50.0f));

		{
			swl::RectangleROI roi(pt1, pt2, true, swl::RectangleROI::real_type(1), swl::RectangleROI::real_type(1), swl::RectangleROI::color_type(), swl::RectangleROI::color_type());
			BOOST_CHECK(!roi.moveVertex(swl::RectangleROI::point_type(-21.0f, 10.0f), delta, limitRegion, 0.1f));
			BOOST_CHECK(local::comparePoints(roi.point1(), pt1));
			BOOST_CHECK(local::comparePoints(roi.point2(), pt2));

			BOOST_CHECK(roi.moveVertex(swl::RectangleROI::point_type(-21.0f, 10.0f), delta, limitRegion, 2.0f));
			BOOST_CHECK(local::comparePoints(roi.point1(), pt1 + delta));
			BOOST_CHECK(local::comparePoints(roi.point2(), pt2));
		}

		{
			swl::RectangleROI roi(pt1, pt2, true, swl::RectangleROI::real_type(1), swl::RectangleROI::real_type(1), swl::RectangleROI::color_type(), swl::RectangleROI::color_type());
			BOOST_CHECK(!roi.moveVertex(swl::RectangleROI::point_type(39.0f, 26.0f), delta, limitRegion, 0.1f));
			BOOST_CHECK(local::comparePoints(roi.point1(), pt1));
			BOOST_CHECK(local::comparePoints(roi.point2(), pt2));

			BOOST_CHECK(roi.moveVertex(swl::RectangleROI::point_type(39.0f, 26.0f), delta, limitRegion, 2.0f));
			BOOST_CHECK(local::comparePoints(roi.point1(), pt1));
			BOOST_CHECK(local::comparePoints(roi.point2(), pt2 + delta));
		}

		{
			swl::RectangleROI roi(pt1, pt2, true, swl::RectangleROI::real_type(1), swl::RectangleROI::real_type(1), swl::RectangleROI::color_type(), swl::RectangleROI::color_type());
			BOOST_CHECK(roi.moveVertex(swl::RectangleROI::point_type(-21.0f, 10.0f), bigDelta, limitRegion, 2.0f));
			BOOST_CHECK(local::comparePoints(roi.point1(), swl::RectangleROI::point_type(limitRegion.right, limitRegion.bottom)));
			BOOST_CHECK(!local::comparePoints(roi.point2(), swl::RectangleROI::point_type(limitRegion.right, limitRegion.bottom)));
			BOOST_CHECK(local::comparePoints(roi.point2(), pt2));
		}

		{
			swl::RectangleROI roi(pt1, pt2, true, swl::RectangleROI::real_type(1), swl::RectangleROI::real_type(1), swl::RectangleROI::color_type(), swl::RectangleROI::color_type());
			BOOST_CHECK(roi.moveVertex(swl::RectangleROI::point_type(39.0f, 26.0f), bigDelta, limitRegion, 2.0f));
			BOOST_CHECK(!local::comparePoints(roi.point1(), swl::RectangleROI::point_type(limitRegion.right, limitRegion.bottom)));
			BOOST_CHECK(local::comparePoints(roi.point1(), pt1));
			BOOST_CHECK(local::comparePoints(roi.point2(), swl::RectangleROI::point_type(limitRegion.right, limitRegion.bottom)));
		}
	}

	void testMoveRegion()
	{
		Fixture fixture;

		const swl::RectangleROI::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f);
		const swl::RectangleROI::point_type delta(3.0f, -7.0f);

		swl::RectangleROI roi(pt1, pt2, true, swl::RectangleROI::real_type(1), swl::RectangleROI::real_type(1), swl::RectangleROI::color_type(), swl::RectangleROI::color_type());
		roi.moveRegion(delta);
		BOOST_CHECK(local::comparePoints(roi.point1(), pt1 + delta));
		BOOST_CHECK(local::comparePoints(roi.point2(), pt2 + delta));
	}

	void testMoveRegionWithLimit()
	{
		Fixture fixture;

		const swl::RectangleROI::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f);
		const swl::RectangleROI::region_type limitRegion(swl::RectangleROI::point_type(-5.0f, -5.0f), swl::RectangleROI::point_type(50.0f, 50.0f));

		{
			const swl::RectangleROI::point_type delta(5.0f, 10.0f);

			swl::RectangleROI roi(pt1, pt2, true, swl::RectangleROI::real_type(1), swl::RectangleROI::real_type(1), swl::RectangleROI::color_type(), swl::RectangleROI::color_type());
			roi.moveRegion(delta, limitRegion);
			BOOST_CHECK(local::comparePoints(roi.point1(), pt1 + delta));
			BOOST_CHECK(local::comparePoints(roi.point2(), pt2 + delta));
		}

		{
			const swl::RectangleROI::point_type bigDelta(100.0f, -100.0f);
			const swl::RectangleROI::real_type dx = 10.0f, dy = -15.0f;  // actual displacement

			swl::RectangleROI roi(pt1, pt2, true, swl::RectangleROI::real_type(1), swl::RectangleROI::real_type(1), swl::RectangleROI::color_type(), swl::RectangleROI::color_type());
			roi.moveRegion(bigDelta, limitRegion);
			BOOST_CHECK(local::comparePoints(roi.point1(), pt1 + swl::RectangleROI::point_type(dx, dy)));  // caution: not (-5, -5), but (-10, -5)
			BOOST_CHECK(local::comparePoints(roi.point2(), pt2 + swl::RectangleROI::point_type(dx, dy)));
		}

		{
			const swl::RectangleROI::point_type delta(-5.0f, 100.0f);
			const swl::RectangleROI::real_type dx = -5.0f, dy = 25.0f;  // computed displacement
			const swl::RectangleROI::real_type dx2 = 0.0f;  // actual displacement

			swl::RectangleROI roi(pt1, pt2, true, swl::RectangleROI::real_type(1), swl::RectangleROI::real_type(1), swl::RectangleROI::color_type(), swl::RectangleROI::color_type());
			roi.moveRegion(delta, limitRegion);
			BOOST_CHECK(!local::comparePoints(roi.point1(), pt1 + swl::RectangleROI::point_type(dx, dy)));  // caution: not (-25, 35), but (-20, 35)  ==>  don't move along x-axis because x-value is beyond a limit region & away from its boundary
			BOOST_CHECK(local::comparePoints(roi.point1(), pt1 + swl::RectangleROI::point_type(dx2, dy)));
			BOOST_CHECK(!local::comparePoints(roi.point2(), pt2 + swl::RectangleROI::point_type(dx, dy)));
			BOOST_CHECK(local::comparePoints(roi.point2(), pt2 + swl::RectangleROI::point_type(dx2, dy)));
		}
	}

	void testIsVertex()
	{
		Fixture fixture;

		const swl::RectangleROI::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f);
		const swl::RectangleROI roi(pt1, pt2, true, swl::RectangleROI::real_type(1), swl::RectangleROI::real_type(1), swl::RectangleROI::color_type(), swl::RectangleROI::color_type());

		BOOST_CHECK(roi.isVertex(swl::RectangleROI::point_type(-20.1f, 10.0f), swl::RectangleROI::real_type(0.5)));
		BOOST_CHECK(!roi.isVertex(swl::RectangleROI::point_type(-20.1f, 10.0f), swl::RectangleROI::real_type(0.05)));
		BOOST_CHECK(roi.isVertex(swl::RectangleROI::point_type(41.0f, 23.5f), swl::RectangleROI::real_type(3)));
		BOOST_CHECK(!roi.isVertex(swl::RectangleROI::point_type(40.1f, 25.3f), swl::RectangleROI::real_type(0.2)));

		BOOST_CHECK(roi.isVertex(swl::RectangleROI::point_type(pt1.x, pt2.y), swl::RectangleROI::real_type(0.01)));
		BOOST_CHECK(roi.isVertex(swl::RectangleROI::point_type(pt2.x, pt1.y), swl::RectangleROI::real_type(0.01)));

		BOOST_CHECK(!roi.isVertex((pt1 + pt2) / swl::RectangleROI::real_type(2), swl::RectangleROI::real_type(1)));
	}

	void testInclude()
	{
		Fixture fixture;

		const swl::RectangleROI::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f);
		const swl::RectangleROI roi(pt1, pt2, true, swl::RectangleROI::real_type(1), swl::RectangleROI::real_type(1), swl::RectangleROI::color_type(), swl::RectangleROI::color_type());

		BOOST_CHECK(!roi.include(swl::RectangleROI::point_type(0, 0), swl::RectangleROI::real_type(0.01)));
		BOOST_CHECK(roi.include(swl::RectangleROI::point_type(30, 20), swl::RectangleROI::real_type(0.01)));
		BOOST_CHECK(roi.include((pt1 + pt2) / swl::RectangleROI::real_type(2), swl::RectangleROI::real_type(0.01)));

		BOOST_CHECK(roi.include(local::calculatePoint(pt1, pt2, swl::RectangleROI::real_type(0.1)), swl::RectangleROI::real_type(0.01)));
		BOOST_CHECK(roi.include(local::calculatePoint(pt1, pt2, swl::RectangleROI::real_type(0.83)), swl::RectangleROI::real_type(0.01)));
		BOOST_CHECK(!roi.include(local::calculatePoint(pt1, pt2, swl::RectangleROI::real_type(2.1)), swl::RectangleROI::real_type(0.01)));
		BOOST_CHECK(!roi.include(local::calculatePoint(pt1, pt2, swl::RectangleROI::real_type(-15.8)), swl::RectangleROI::real_type(0.01)));
	}
};

struct RectangleROITestSuite: public boost::unit_test_framework::test_suite
{
	RectangleROITestSuite()
	: boost::unit_test_framework::test_suite("SWL.Util.RectangleROI")
	{
		boost::shared_ptr<RectangleROITest> test(new RectangleROITest());

		add(BOOST_CLASS_TEST_CASE(&RectangleROITest::testMoveVertex, test), 0);
		add(BOOST_CLASS_TEST_CASE(&RectangleROITest::testMoveVertexWithLimit, test), 0);
		add(BOOST_CLASS_TEST_CASE(&RectangleROITest::testMoveRegion, test), 0);
		add(BOOST_CLASS_TEST_CASE(&RectangleROITest::testMoveRegionWithLimit, test), 0);
		add(BOOST_CLASS_TEST_CASE(&RectangleROITest::testIsVertex, test), 0);
		add(BOOST_CLASS_TEST_CASE(&RectangleROITest::testInclude, test), 0);

		boost::unit_test::framework::master_test_suite().add(this);
	}
} testsuite;

}  // unnamed namespace

//-----------------------------------------------------------------------------
// Google Test

#elif defined(__SWL_UNIT_TEST__USE_GOOGLE_TEST)

class RectangleROITest : public testing::Test
{
protected:
	/*virtual*/ void SetUp()
	{
	}

	/*virtual*/ void TearDown()
	{
	}
};

TEST_F(RectangleROITest, testMoveVertex)
{
	const swl::RectangleROI::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f);
	const swl::RectangleROI::point_type delta(3.0f, -7.0f);

	{
		swl::RectangleROI roi(pt1, pt2, true, swl::RectangleROI::real_type(1), swl::RectangleROI::real_type(1), swl::RectangleROI::color_type(), swl::RectangleROI::color_type());
		EXPECT_FALSE(roi.moveVertex(swl::RectangleROI::point_type(-21.0f, 10.0f), delta, 0.1f));
		EXPECT_TRUE(local::comparePoints(roi.point1(), pt1));
		EXPECT_TRUE(local::comparePoints(roi.point2(), pt2));

		EXPECT_TRUE(roi.moveVertex(swl::RectangleROI::point_type(-21.0f, 10.0f), delta, 2.0f));
		EXPECT_TRUE(local::comparePoints(roi.point1(), pt1 + delta));
		EXPECT_FALSE(local::comparePoints(roi.point2(), pt2 + delta));
		EXPECT_TRUE(local::comparePoints(roi.point2(), pt2));
	}

	{
		swl::RectangleROI roi(pt1, pt2, true, swl::RectangleROI::real_type(1), swl::RectangleROI::real_type(1), swl::RectangleROI::color_type(), swl::RectangleROI::color_type());
		EXPECT_FALSE(roi.moveVertex(swl::RectangleROI::point_type(39.0f, 26.0f), delta, 0.1f));
		EXPECT_TRUE(local::comparePoints(roi.point1(), pt1));
		EXPECT_TRUE(local::comparePoints(roi.point2(), pt2));

		EXPECT_TRUE(roi.moveVertex(swl::RectangleROI::point_type(39.0f, 26.0f), delta, 2.0f));
		EXPECT_FALSE(local::comparePoints(roi.point1(), pt1 + delta));
		EXPECT_TRUE(local::comparePoints(roi.point1(), pt1));
		EXPECT_TRUE(local::comparePoints(roi.point2(), pt2 + delta));
	}
}

TEST_F(RectangleROITest, testMoveVertexWithLimit)
{
	const swl::RectangleROI::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f);
	const swl::RectangleROI::point_type delta(5.0f, 10.0f);
	const swl::RectangleROI::point_type bigDelta(100.0f, -100.0f);
	const swl::RectangleROI::region_type limitRegion(swl::RectangleROI::point_type(-5.0f, -5.0f), swl::RectangleROI::point_type(50.0f, 50.0f));

	{
		swl::RectangleROI roi(pt1, pt2, true, swl::RectangleROI::real_type(1), swl::RectangleROI::real_type(1), swl::RectangleROI::color_type(), swl::RectangleROI::color_type());
		EXPECT_FALSE(roi.moveVertex(swl::RectangleROI::point_type(-21.0f, 10.0f), delta, limitRegion, 0.1f));
		EXPECT_TRUE(local::comparePoints(roi.point1(), pt1));
		EXPECT_TRUE(local::comparePoints(roi.point2(), pt2));

		EXPECT_TRUE(roi.moveVertex(swl::RectangleROI::point_type(-21.0f, 10.0f), delta, limitRegion, 2.0f));
		EXPECT_TRUE(local::comparePoints(roi.point1(), pt1 + delta));
		EXPECT_TRUE(local::comparePoints(roi.point2(), pt2));
	}

	{
		swl::RectangleROI roi(pt1, pt2, true, swl::RectangleROI::real_type(1), swl::RectangleROI::real_type(1), swl::RectangleROI::color_type(), swl::RectangleROI::color_type());
		EXPECT_FALSE(roi.moveVertex(swl::RectangleROI::point_type(39.0f, 26.0f), delta, limitRegion, 0.1f));
		EXPECT_TRUE(local::comparePoints(roi.point1(), pt1));
		EXPECT_TRUE(local::comparePoints(roi.point2(), pt2));

		EXPECT_TRUE(roi.moveVertex(swl::RectangleROI::point_type(39.0f, 26.0f), delta, limitRegion, 2.0f));
		EXPECT_TRUE(local::comparePoints(roi.point1(), pt1));
		EXPECT_TRUE(local::comparePoints(roi.point2(), pt2 + delta));
	}

	{
		swl::RectangleROI roi(pt1, pt2, true, swl::RectangleROI::real_type(1), swl::RectangleROI::real_type(1), swl::RectangleROI::color_type(), swl::RectangleROI::color_type());
		EXPECT_TRUE(roi.moveVertex(swl::RectangleROI::point_type(-21.0f, 10.0f), bigDelta, limitRegion, 2.0f));
		EXPECT_TRUE(local::comparePoints(roi.point1(), swl::RectangleROI::point_type(limitRegion.right, limitRegion.bottom)));
		EXPECT_FALSE(local::comparePoints(roi.point2(), swl::RectangleROI::point_type(limitRegion.right, limitRegion.bottom)));
		EXPECT_TRUE(local::comparePoints(roi.point2(), pt2));
	}

	{
		swl::RectangleROI roi(pt1, pt2, true, swl::RectangleROI::real_type(1), swl::RectangleROI::real_type(1), swl::RectangleROI::color_type(), swl::RectangleROI::color_type());
		EXPECT_TRUE(roi.moveVertex(swl::RectangleROI::point_type(39.0f, 26.0f), bigDelta, limitRegion, 2.0f));
		EXPECT_FALSE(local::comparePoints(roi.point1(), swl::RectangleROI::point_type(limitRegion.right, limitRegion.bottom)));
		EXPECT_TRUE(local::comparePoints(roi.point1(), pt1));
		EXPECT_TRUE(local::comparePoints(roi.point2(), swl::RectangleROI::point_type(limitRegion.right, limitRegion.bottom)));
	}
}

TEST_F(RectangleROITest, testMoveRegion)
{
	const swl::RectangleROI::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f);
	const swl::RectangleROI::point_type delta(3.0f, -7.0f);

	swl::RectangleROI roi(pt1, pt2, true, swl::RectangleROI::real_type(1), swl::RectangleROI::real_type(1), swl::RectangleROI::color_type(), swl::RectangleROI::color_type());
	roi.moveRegion(delta);
	EXPECT_TRUE(local::comparePoints(roi.point1(), pt1 + delta));
	EXPECT_TRUE(local::comparePoints(roi.point2(), pt2 + delta));
}

TEST_F(RectangleROITest, testMoveRegionWithLimit)
{
	const swl::RectangleROI::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f);
	const swl::RectangleROI::region_type limitRegion(swl::RectangleROI::point_type(-5.0f, -5.0f), swl::RectangleROI::point_type(50.0f, 50.0f));

	{
		const swl::RectangleROI::point_type delta(5.0f, 10.0f);

		swl::RectangleROI roi(pt1, pt2, true, swl::RectangleROI::real_type(1), swl::RectangleROI::real_type(1), swl::RectangleROI::color_type(), swl::RectangleROI::color_type());
		roi.moveRegion(delta, limitRegion);
		EXPECT_TRUE(local::comparePoints(roi.point1(), pt1 + delta));
		EXPECT_TRUE(local::comparePoints(roi.point2(), pt2 + delta));
	}

	{
		const swl::RectangleROI::point_type bigDelta(100.0f, -100.0f);
		const swl::RectangleROI::real_type dx = 10.0f, dy = -15.0f;  // actual displacement

		swl::RectangleROI roi(pt1, pt2, true, swl::RectangleROI::real_type(1), swl::RectangleROI::real_type(1), swl::RectangleROI::color_type(), swl::RectangleROI::color_type());
		roi.moveRegion(bigDelta, limitRegion);
		EXPECT_TRUE(local::comparePoints(roi.point1(), pt1 + swl::RectangleROI::point_type(dx, dy)));  // caution: not (-5, -5), but (-10, -5)
		EXPECT_TRUE(local::comparePoints(roi.point2(), pt2 + swl::RectangleROI::point_type(dx, dy)));
	}

	{
		const swl::RectangleROI::point_type delta(-5.0f, 100.0f);
		const swl::RectangleROI::real_type dx = -5.0f, dy = 25.0f;  // computed displacement
		const swl::RectangleROI::real_type dx2 = 0.0f;  // actual displacement

		swl::RectangleROI roi(pt1, pt2, true, swl::RectangleROI::real_type(1), swl::RectangleROI::real_type(1), swl::RectangleROI::color_type(), swl::RectangleROI::color_type());
		roi.moveRegion(delta, limitRegion);
		EXPECT_FALSE(local::comparePoints(roi.point1(), pt1 + swl::RectangleROI::point_type(dx, dy)));  // caution: not (-25, 35), but (-20, 35)  ==>  don't move along x-axis because x-value is beyond a limit region & away from its boundary
		EXPECT_TRUE(local::comparePoints(roi.point1(), pt1 + swl::RectangleROI::point_type(dx2, dy)));
		EXPECT_TRUE(!local::comparePoints(roi.point2(), pt2 + swl::RectangleROI::point_type(dx, dy)));
		EXPECT_TRUE(local::comparePoints(roi.point2(), pt2 + swl::RectangleROI::point_type(dx2, dy)));
	}
}

TEST_F(RectangleROITest, testIsVertex)
{
	const swl::RectangleROI::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f);
	const swl::RectangleROI roi(pt1, pt2, true, swl::RectangleROI::real_type(1), swl::RectangleROI::real_type(1), swl::RectangleROI::color_type(), swl::RectangleROI::color_type());

	EXPECT_TRUE(roi.isVertex(swl::RectangleROI::point_type(-20.1f, 10.0f), swl::RectangleROI::real_type(0.5)));
	EXPECT_FALSE(roi.isVertex(swl::RectangleROI::point_type(-20.1f, 10.0f), swl::RectangleROI::real_type(0.05)));
	EXPECT_TRUE(roi.isVertex(swl::RectangleROI::point_type(41.0f, 23.5f), swl::RectangleROI::real_type(3)));
	EXPECT_FALSE(roi.isVertex(swl::RectangleROI::point_type(40.1f, 25.3f), swl::RectangleROI::real_type(0.2)));

	EXPECT_TRUE(roi.isVertex(swl::RectangleROI::point_type(pt1.x, pt2.y), swl::RectangleROI::real_type(0.01)));
	EXPECT_TRUE(roi.isVertex(swl::RectangleROI::point_type(pt2.x, pt1.y), swl::RectangleROI::real_type(0.01)));

	EXPECT_FALSE(roi.isVertex((pt1 + pt2) / swl::RectangleROI::real_type(2), swl::RectangleROI::real_type(1)));
}

TEST_F(RectangleROITest, testInclude)
{
	const swl::RectangleROI::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f);
	const swl::RectangleROI roi(pt1, pt2, true, swl::RectangleROI::real_type(1), swl::RectangleROI::real_type(1), swl::RectangleROI::color_type(), swl::RectangleROI::color_type());

	EXPECT_FALSE(roi.include(swl::RectangleROI::point_type(0, 0), swl::RectangleROI::real_type(0.01)));
	EXPECT_TRUE(roi.include(swl::RectangleROI::point_type(30, 20), swl::RectangleROI::real_type(0.01)));
	EXPECT_TRUE(roi.include((pt1 + pt2) / swl::RectangleROI::real_type(2), swl::RectangleROI::real_type(0.01)));

	EXPECT_TRUE(roi.include(local::calculatePoint(pt1, pt2, swl::RectangleROI::real_type(0.1)), swl::RectangleROI::real_type(0.01)));
	EXPECT_TRUE(roi.include(local::calculatePoint(pt1, pt2, swl::RectangleROI::real_type(0.83)), swl::RectangleROI::real_type(0.01)));
	EXPECT_FALSE(roi.include(local::calculatePoint(pt1, pt2, swl::RectangleROI::real_type(2.1)), swl::RectangleROI::real_type(0.01)));
	EXPECT_FALSE(roi.include(local::calculatePoint(pt1, pt2, swl::RectangleROI::real_type(-15.8)), swl::RectangleROI::real_type(0.01)));
}

//-----------------------------------------------------------------------------
// CppUnit

#elif defined(__SWL_UNIT_TEST__USE_CPP_UNIT)

struct RectangleROITest: public CppUnit::TestFixture
{
private:
	CPPUNIT_TEST_SUITE(RectangleROITest);
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
		const swl::RectangleROI::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f);
		const swl::RectangleROI::point_type delta(3.0f, -7.0f);

		{
			swl::RectangleROI roi(pt1, pt2, true, swl::RectangleROI::real_type(1), swl::RectangleROI::real_type(1), swl::RectangleROI::color_type(), swl::RectangleROI::color_type());
			CPPUNIT_ASSERT(!roi.moveVertex(swl::RectangleROI::point_type(-21.0f, 10.0f), delta, 0.1f));
			CPPUNIT_ASSERT(local::comparePoints(roi.point1(), pt1));
			CPPUNIT_ASSERT(local::comparePoints(roi.point2(), pt2));

			CPPUNIT_ASSERT(roi.moveVertex(swl::RectangleROI::point_type(-21.0f, 10.0f), delta, 2.0f));
			CPPUNIT_ASSERT(local::comparePoints(roi.point1(), pt1 + delta));
			CPPUNIT_ASSERT(!local::comparePoints(roi.point2(), pt2 + delta));
			CPPUNIT_ASSERT(local::comparePoints(roi.point2(), pt2));
		}

		{
			swl::RectangleROI roi(pt1, pt2, true, swl::RectangleROI::real_type(1), swl::RectangleROI::real_type(1), swl::RectangleROI::color_type(), swl::RectangleROI::color_type());
			CPPUNIT_ASSERT(!roi.moveVertex(swl::RectangleROI::point_type(39.0f, 26.0f), delta, 0.1f));
			CPPUNIT_ASSERT(local::comparePoints(roi.point1(), pt1));
			CPPUNIT_ASSERT(local::comparePoints(roi.point2(), pt2));

			CPPUNIT_ASSERT(roi.moveVertex(swl::RectangleROI::point_type(39.0f, 26.0f), delta, 2.0f));
			CPPUNIT_ASSERT(!local::comparePoints(roi.point1(), pt1 + delta));
			CPPUNIT_ASSERT(local::comparePoints(roi.point1(), pt1));
			CPPUNIT_ASSERT(local::comparePoints(roi.point2(), pt2 + delta));
		}
	}

	void testMoveVertexWithLimit()
	{
		const swl::RectangleROI::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f);
		const swl::RectangleROI::point_type delta(5.0f, 10.0f);
		const swl::RectangleROI::point_type bigDelta(100.0f, -100.0f);
		const swl::RectangleROI::region_type limitRegion(swl::RectangleROI::point_type(-5.0f, -5.0f), swl::RectangleROI::point_type(50.0f, 50.0f));

		{
			swl::RectangleROI roi(pt1, pt2, true, swl::RectangleROI::real_type(1), swl::RectangleROI::real_type(1), swl::RectangleROI::color_type(), swl::RectangleROI::color_type());
			CPPUNIT_ASSERT(!roi.moveVertex(swl::RectangleROI::point_type(-21.0f, 10.0f), delta, limitRegion, 0.1f));
			CPPUNIT_ASSERT(local::comparePoints(roi.point1(), pt1));
			CPPUNIT_ASSERT(local::comparePoints(roi.point2(), pt2));

			CPPUNIT_ASSERT(roi.moveVertex(swl::RectangleROI::point_type(-21.0f, 10.0f), delta, limitRegion, 2.0f));
			CPPUNIT_ASSERT(local::comparePoints(roi.point1(), pt1 + delta));
			CPPUNIT_ASSERT(local::comparePoints(roi.point2(), pt2));
		}

		{
			swl::RectangleROI roi(pt1, pt2, true, swl::RectangleROI::real_type(1), swl::RectangleROI::real_type(1), swl::RectangleROI::color_type(), swl::RectangleROI::color_type());
			CPPUNIT_ASSERT(!roi.moveVertex(swl::RectangleROI::point_type(39.0f, 26.0f), delta, limitRegion, 0.1f));
			CPPUNIT_ASSERT(local::comparePoints(roi.point1(), pt1));
			CPPUNIT_ASSERT(local::comparePoints(roi.point2(), pt2));

			CPPUNIT_ASSERT(roi.moveVertex(swl::RectangleROI::point_type(39.0f, 26.0f), delta, limitRegion, 2.0f));
			CPPUNIT_ASSERT(local::comparePoints(roi.point1(), pt1));
			CPPUNIT_ASSERT(local::comparePoints(roi.point2(), pt2 + delta));
		}

		{
			swl::RectangleROI roi(pt1, pt2, true, swl::RectangleROI::real_type(1), swl::RectangleROI::real_type(1), swl::RectangleROI::color_type(), swl::RectangleROI::color_type());
			CPPUNIT_ASSERT(roi.moveVertex(swl::RectangleROI::point_type(-21.0f, 10.0f), bigDelta, limitRegion, 2.0f));
			CPPUNIT_ASSERT(local::comparePoints(roi.point1(), swl::RectangleROI::point_type(limitRegion.right, limitRegion.bottom)));
			CPPUNIT_ASSERT(!local::comparePoints(roi.point2(), swl::RectangleROI::point_type(limitRegion.right, limitRegion.bottom)));
			CPPUNIT_ASSERT(local::comparePoints(roi.point2(), pt2));
		}

		{
			swl::RectangleROI roi(pt1, pt2, true, swl::RectangleROI::real_type(1), swl::RectangleROI::real_type(1), swl::RectangleROI::color_type(), swl::RectangleROI::color_type());
			CPPUNIT_ASSERT(roi.moveVertex(swl::RectangleROI::point_type(39.0f, 26.0f), bigDelta, limitRegion, 2.0f));
			CPPUNIT_ASSERT(!local::comparePoints(roi.point1(), swl::RectangleROI::point_type(limitRegion.right, limitRegion.bottom)));
			CPPUNIT_ASSERT(local::comparePoints(roi.point1(), pt1));
			CPPUNIT_ASSERT(local::comparePoints(roi.point2(), swl::RectangleROI::point_type(limitRegion.right, limitRegion.bottom)));
		}
	}

	void testMoveRegion()
	{
		const swl::RectangleROI::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f);
		const swl::RectangleROI::point_type delta(3.0f, -7.0f);

		swl::RectangleROI roi(pt1, pt2, true, swl::RectangleROI::real_type(1), swl::RectangleROI::real_type(1), swl::RectangleROI::color_type(), swl::RectangleROI::color_type());
		roi.moveRegion(delta);
		CPPUNIT_ASSERT(local::comparePoints(roi.point1(), pt1 + delta));
		CPPUNIT_ASSERT(local::comparePoints(roi.point2(), pt2 + delta));
	}

	void testMoveRegionWithLimit()
	{
		const swl::RectangleROI::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f);
		const swl::RectangleROI::region_type limitRegion(swl::RectangleROI::point_type(-5.0f, -5.0f), swl::RectangleROI::point_type(50.0f, 50.0f));

		{
			const swl::RectangleROI::point_type delta(5.0f, 10.0f);

			swl::RectangleROI roi(pt1, pt2, true, swl::RectangleROI::real_type(1), swl::RectangleROI::real_type(1), swl::RectangleROI::color_type(), swl::RectangleROI::color_type());
			roi.moveRegion(delta, limitRegion);
			CPPUNIT_ASSERT(local::comparePoints(roi.point1(), pt1 + delta));
			CPPUNIT_ASSERT(local::comparePoints(roi.point2(), pt2 + delta));
		}

		{
			const swl::RectangleROI::point_type bigDelta(100.0f, -100.0f);
			const swl::RectangleROI::real_type dx = 10.0f, dy = -15.0f;  // actual displacement

			swl::RectangleROI roi(pt1, pt2, true, swl::RectangleROI::real_type(1), swl::RectangleROI::real_type(1), swl::RectangleROI::color_type(), swl::RectangleROI::color_type());
			roi.moveRegion(bigDelta, limitRegion);
			CPPUNIT_ASSERT(local::comparePoints(roi.point1(), pt1 + swl::RectangleROI::point_type(dx, dy)));  // caution: not (-5, -5), but (-10, -5)
			CPPUNIT_ASSERT(local::comparePoints(roi.point2(), pt2 + swl::RectangleROI::point_type(dx, dy)));
		}

		{
			const swl::RectangleROI::point_type delta(-5.0f, 100.0f);
			const swl::RectangleROI::real_type dx = -5.0f, dy = 25.0f;  // computed displacement
			const swl::RectangleROI::real_type dx2 = 0.0f;  // actual displacement

			swl::RectangleROI roi(pt1, pt2, true, swl::RectangleROI::real_type(1), swl::RectangleROI::real_type(1), swl::RectangleROI::color_type(), swl::RectangleROI::color_type());
			roi.moveRegion(delta, limitRegion);
			CPPUNIT_ASSERT(!local::comparePoints(roi.point1(), pt1 + swl::RectangleROI::point_type(dx, dy)));  // caution: not (-25, 35), but (-20, 35)  ==>  don't move along x-axis because x-value is beyond a limit region & away from its boundary
			CPPUNIT_ASSERT(local::comparePoints(roi.point1(), pt1 + swl::RectangleROI::point_type(dx2, dy)));
			CPPUNIT_ASSERT(!local::comparePoints(roi.point2(), pt2 + swl::RectangleROI::point_type(dx, dy)));
			CPPUNIT_ASSERT(local::comparePoints(roi.point2(), pt2 + swl::RectangleROI::point_type(dx2, dy)));
		}
	}

	void testIsVertex()
	{
		const swl::RectangleROI::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f);
		const swl::RectangleROI roi(pt1, pt2, true, swl::RectangleROI::real_type(1), swl::RectangleROI::real_type(1), swl::RectangleROI::color_type(), swl::RectangleROI::color_type());

		CPPUNIT_ASSERT(roi.isVertex(swl::RectangleROI::point_type(-20.1f, 10.0f), swl::RectangleROI::real_type(0.5)));
		CPPUNIT_ASSERT(!roi.isVertex(swl::RectangleROI::point_type(-20.1f, 10.0f), swl::RectangleROI::real_type(0.05)));
		CPPUNIT_ASSERT(roi.isVertex(swl::RectangleROI::point_type(41.0f, 23.5f), swl::RectangleROI::real_type(3)));
		CPPUNIT_ASSERT(!roi.isVertex(swl::RectangleROI::point_type(40.1f, 25.3f), swl::RectangleROI::real_type(0.2)));

		CPPUNIT_ASSERT(roi.isVertex(swl::RectangleROI::point_type(pt1.x, pt2.y), swl::RectangleROI::real_type(0.01)));
		CPPUNIT_ASSERT(roi.isVertex(swl::RectangleROI::point_type(pt2.x, pt1.y), swl::RectangleROI::real_type(0.01)));

		CPPUNIT_ASSERT(!roi.isVertex((pt1 + pt2) / swl::RectangleROI::real_type(2), swl::RectangleROI::real_type(1)));
	}

	void testInclude()
	{
		const swl::RectangleROI::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f);
		const swl::RectangleROI roi(pt1, pt2, true, swl::RectangleROI::real_type(1), swl::RectangleROI::real_type(1), swl::RectangleROI::color_type(), swl::RectangleROI::color_type());

		CPPUNIT_ASSERT(!roi.include(swl::RectangleROI::point_type(0, 0), swl::RectangleROI::real_type(0.01)));
		CPPUNIT_ASSERT(roi.include(swl::RectangleROI::point_type(30, 20), swl::RectangleROI::real_type(0.01)));
		CPPUNIT_ASSERT(roi.include((pt1 + pt2) / swl::RectangleROI::real_type(2), swl::RectangleROI::real_type(0.01)));

		CPPUNIT_ASSERT(roi.include(local::calculatePoint(pt1, pt2, swl::RectangleROI::real_type(0.1)), swl::RectangleROI::real_type(0.01)));
		CPPUNIT_ASSERT(roi.include(local::calculatePoint(pt1, pt2, swl::RectangleROI::real_type(0.83)), swl::RectangleROI::real_type(0.01)));
		CPPUNIT_ASSERT(!roi.include(local::calculatePoint(pt1, pt2, swl::RectangleROI::real_type(2.1)), swl::RectangleROI::real_type(0.01)));
		CPPUNIT_ASSERT(!roi.include(local::calculatePoint(pt1, pt2, swl::RectangleROI::real_type(-15.8)), swl::RectangleROI::real_type(0.01)));
	}
};

#endif

}  // namespace unit_test
}  // namespace swl

#if defined(__SWL_UNIT_TEST__USE_CPP_UNIT)
//CPPUNIT_TEST_SUITE_REGISTRATION(swl::unit_test::RectangleROITest);
CPPUNIT_REGISTRY_ADD_TO_DEFAULT("SWL.Util");
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(swl::unit_test::RectangleROITest, "SWL.Util");
#endif
