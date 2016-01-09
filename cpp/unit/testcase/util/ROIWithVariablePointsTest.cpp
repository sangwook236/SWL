#include "swl/Config.h"
#include "../../UnitTestConfig.h"
#include "swl/base/String.h"
#include "swl/util/RegionOfInterest.h"
#include "swl/base/LogException.h"
#include <boost/smart_ptr.hpp>
#include <cmath>
#include <ctime>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace {
namespace local {

bool comparePoints(const swl::ROIWithVariablePoints::point_type &lhs, const swl::ROIWithVariablePoints::point_type &rhs, const swl::ROIWithVariablePoints::real_type &eps = swl::ROIWithVariablePoints::real_type(1.0e-15))
{
	return std::abs(lhs.x - rhs.x) <= eps && std::abs(lhs.y - rhs.y) <= eps;
}

}  // namespace local
}  // unnamed namespace

namespace swl {
namespace unit_test {

//-----------------------------------------------------------------------------
// Boost Test

#if defined(__SWL_UNIT_TEST__USE_BOOST_TEST)

namespace {

struct ROIWithVariablePointsTest
{
private:
	struct Fixture
	{
		Fixture()  // set up
		{
			std::srand((unsigned int)std::time(NULL));
		}

		~Fixture()  // tear down
		{
		}
	};

public:
	void testHandlePoint()
	{
		Fixture fixture;

		boost::scoped_ptr<swl::ROIWithVariablePoints> roi(new swl::PolylineROI(true, swl::PolylineROI::real_type(1), swl::PolylineROI::real_type(1), swl::PolylineROI::color_type(), swl::PolylineROI::color_type()));
		BOOST_CHECK(roi);

		BOOST_CHECK(!roi->containPoint());

		const swl::ROIWithVariablePoints::point_type pt1((swl::ROIWithVariablePoints::real_type)std::rand(), (swl::ROIWithVariablePoints::real_type)std::rand());
		roi->addPoint(pt1);
		const swl::ROIWithVariablePoints::point_type pt2((swl::ROIWithVariablePoints::real_type)std::rand(), (swl::ROIWithVariablePoints::real_type)std::rand());
		roi->addPoint(pt2);
		const swl::ROIWithVariablePoints::point_type pt3((swl::ROIWithVariablePoints::real_type)std::rand(), (swl::ROIWithVariablePoints::real_type)std::rand());
		roi->addPoint(pt3);
		const swl::ROIWithVariablePoints::point_type pt4((swl::ROIWithVariablePoints::real_type)std::rand(), (swl::ROIWithVariablePoints::real_type)std::rand());
		roi->addPoint(pt4);
		roi->addPoint(pt4);  // add the same point
		const swl::ROIWithVariablePoints::point_type pt5((swl::ROIWithVariablePoints::real_type)std::rand(), (swl::ROIWithVariablePoints::real_type)std::rand());
		roi->addPoint(pt5);

		BOOST_CHECK(roi->containPoint());
		BOOST_CHECK_EQUAL((std::size_t)6, roi->countPoint());  // not 5

		BOOST_CHECK(local::comparePoints(roi->getPoint(0), pt1));
		BOOST_CHECK(local::comparePoints(roi->getPoint(1), pt2));
		BOOST_CHECK(local::comparePoints(roi->getPoint(2), pt3));
		BOOST_CHECK(local::comparePoints(roi->getPoint(3), pt4));
		BOOST_CHECK(local::comparePoints(roi->getPoint(4), pt4));
		BOOST_CHECK(local::comparePoints(roi->getPoint(5), pt5));

		BOOST_CHECK_THROW(roi->getPoint(6), swl::LogException);

		roi->removePoint(pt4);
		BOOST_CHECK_EQUAL((std::size_t)4, roi->countPoint());  // remove 2 points
		roi->removePoint(pt3);
		BOOST_CHECK_EQUAL((std::size_t)3, roi->countPoint());

		BOOST_CHECK(local::comparePoints(roi->getPoint(0), pt1));
		BOOST_CHECK(local::comparePoints(roi->getPoint(1), pt2));
		BOOST_CHECK(local::comparePoints(roi->getPoint(2), pt5));

		roi->clearAllPoints();
		BOOST_CHECK(!roi->containPoint());
		BOOST_CHECK_EQUAL((std::size_t)0, roi->countPoint());
	}

	void testMoveVertex()
	{
		Fixture fixture;

		const swl::ROIWithVariablePoints::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f), pt3(10.0f, 30.0f), pt4(21.0f, 25.0f), pt5(28.0f, 3.5f);
		const swl::ROIWithVariablePoints::point_type delta(3.0f, -7.0f);

		boost::scoped_ptr<swl::ROIWithVariablePoints> roi(new swl::PolylineROI(true, swl::PolylineROI::real_type(1), swl::PolylineROI::real_type(1), swl::PolylineROI::color_type(), swl::PolylineROI::color_type()));
		BOOST_CHECK(roi);

		roi->addPoint(pt1);
		roi->addPoint(pt2);
		roi->addPoint(pt3);
		roi->addPoint(pt4);
		roi->addPoint(pt5);

		{
			BOOST_CHECK(!roi->moveVertex(swl::ROIWithVariablePoints::point_type(-21.0f, 10.0f), delta, 0.1f));
			BOOST_CHECK(local::comparePoints(roi->getPoint(0), pt1));
			BOOST_CHECK(local::comparePoints(roi->getPoint(1), pt2));

			BOOST_CHECK(roi->moveVertex(swl::ROIWithVariablePoints::point_type(-21.0f, 10.0f), delta, 2.0f));
			BOOST_CHECK(!local::comparePoints(roi->getPoint(0), pt1));
			BOOST_CHECK(local::comparePoints(roi->getPoint(0), pt1 + delta));
			BOOST_CHECK(local::comparePoints(roi->getPoint(1), pt2));
			BOOST_CHECK(!local::comparePoints(roi->getPoint(1), pt2 + delta));
		}

		{
			BOOST_CHECK(!roi->moveVertex(swl::ROIWithVariablePoints::point_type(39.0f, 26.0f), delta, 0.1f));
			BOOST_CHECK(local::comparePoints(roi->getPoint(0), pt1 + delta));
			BOOST_CHECK(local::comparePoints(roi->getPoint(1), pt2));

			BOOST_CHECK(roi->moveVertex(swl::ROIWithVariablePoints::point_type(39.0f, 26.0f), delta, 2.0f));
			BOOST_CHECK(local::comparePoints(roi->getPoint(0), pt1 + delta));
			BOOST_CHECK(!local::comparePoints(roi->getPoint(1), pt2));
			BOOST_CHECK(local::comparePoints(roi->getPoint(1), pt2 + delta));
		}
	}

	void testMoveVertexWithLimit()
	{
		Fixture fixture;

		const swl::ROIWithVariablePoints::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f), pt3(10.0f, 30.0f), pt4(21.0f, 25.0f), pt5(28.0f, 3.5f);
		const swl::ROIWithVariablePoints::point_type delta(5.0f, 10.0f);
		const swl::ROIWithVariablePoints::point_type bigDelta(100.0f, -100.0f);
		const swl::ROIWithVariablePoints::region_type limitRegion(swl::ROIWithVariablePoints::point_type(-5.0f, -5.0f), swl::ROIWithVariablePoints::point_type(50.0f, 50.0f));

		boost::scoped_ptr<swl::ROIWithVariablePoints> roi(new swl::PolylineROI(true, swl::PolylineROI::real_type(1), swl::PolylineROI::real_type(1), swl::PolylineROI::color_type(), swl::PolylineROI::color_type()));
		BOOST_CHECK(roi);

		{
			roi->clearAllPoints();
			roi->addPoint(pt1);
			roi->addPoint(pt2);
			roi->addPoint(pt3);
			roi->addPoint(pt4);
			roi->addPoint(pt5);

			BOOST_CHECK(!roi->moveVertex(swl::ROIWithVariablePoints::point_type(-21.0f, 10.0f), delta, limitRegion, 0.1f));
			BOOST_CHECK(local::comparePoints(roi->getPoint(0), pt1));
			BOOST_CHECK(local::comparePoints(roi->getPoint(1), pt2));

			BOOST_CHECK(roi->moveVertex(swl::ROIWithVariablePoints::point_type(-21.0f, 10.0f), delta, limitRegion, 2.0f));
			BOOST_CHECK(local::comparePoints(roi->getPoint(0), pt1 + delta));
			BOOST_CHECK(local::comparePoints(roi->getPoint(1), pt2));
		}

		{
			roi->clearAllPoints();
			roi->addPoint(pt1);
			roi->addPoint(pt2);
			roi->addPoint(pt3);
			roi->addPoint(pt4);
			roi->addPoint(pt5);

			BOOST_CHECK(!roi->moveVertex(swl::ROIWithVariablePoints::point_type(39.0f, 26.0f), delta, limitRegion, 0.1f));
			BOOST_CHECK(local::comparePoints(roi->getPoint(0), pt1));
			BOOST_CHECK(local::comparePoints(roi->getPoint(1), pt2));

			BOOST_CHECK(roi->moveVertex(swl::ROIWithVariablePoints::point_type(39.0f, 26.0f), delta, limitRegion, 2.0f));
			BOOST_CHECK(local::comparePoints(roi->getPoint(0), pt1));
			BOOST_CHECK(local::comparePoints(roi->getPoint(1), pt2 + delta));
		}

		{
			roi->clearAllPoints();
			roi->addPoint(pt1);
			roi->addPoint(pt2);
			roi->addPoint(pt3);
			roi->addPoint(pt4);
			roi->addPoint(pt5);

			BOOST_CHECK(roi->moveVertex(swl::ROIWithVariablePoints::point_type(-21.0f, 10.0f), bigDelta, limitRegion, 2.0f));
			BOOST_CHECK(local::comparePoints(roi->getPoint(0), swl::ROIWithVariablePoints::point_type(limitRegion.right, limitRegion.bottom)));
			BOOST_CHECK(!local::comparePoints(roi->getPoint(1), swl::ROIWithVariablePoints::point_type(limitRegion.right, limitRegion.bottom)));
			BOOST_CHECK(local::comparePoints(roi->getPoint(1), pt2));
		}

		{
			roi->clearAllPoints();
			roi->addPoint(pt1);
			roi->addPoint(pt2);
			roi->addPoint(pt3);
			roi->addPoint(pt4);
			roi->addPoint(pt5);

			BOOST_CHECK(roi->moveVertex(swl::ROIWithVariablePoints::point_type(39.0f, 26.0f), bigDelta, limitRegion, 2.0f));
			BOOST_CHECK(!local::comparePoints(roi->getPoint(0), swl::ROIWithVariablePoints::point_type(limitRegion.right, limitRegion.bottom)));
			BOOST_CHECK(local::comparePoints(roi->getPoint(0), pt1));
			BOOST_CHECK(local::comparePoints(roi->getPoint(1), swl::ROIWithVariablePoints::point_type(limitRegion.right, limitRegion.bottom)));
		}
	}

	void testMoveRegion()
	{
		Fixture fixture;

		const swl::ROIWithVariablePoints::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f), pt3(10.0f, 30.0f), pt4(21.0f, 25.0f), pt5(28.0f, 3.5f);
		const swl::ROIWithVariablePoints::point_type delta(3.0f, -7.0f);

		boost::scoped_ptr<swl::ROIWithVariablePoints> roi(new swl::PolylineROI(true, swl::PolylineROI::real_type(1), swl::PolylineROI::real_type(1), swl::PolylineROI::color_type(), swl::PolylineROI::color_type()));
		BOOST_CHECK(roi);

		roi->addPoint(pt1);
		roi->addPoint(pt2);
		roi->addPoint(pt3);
		roi->addPoint(pt4);
		roi->addPoint(pt5);

		roi->moveRegion(delta);
		BOOST_CHECK(local::comparePoints(roi->getPoint(0), pt1 + delta));
		BOOST_CHECK(local::comparePoints(roi->getPoint(1), pt2 + delta));
		BOOST_CHECK(local::comparePoints(roi->getPoint(2), pt3 + delta));
		BOOST_CHECK(local::comparePoints(roi->getPoint(3), pt4 + delta));
		BOOST_CHECK(local::comparePoints(roi->getPoint(4), pt5 + delta));
	}

	void testMoveRegionWithLimit()
	{
		Fixture fixture;

		const swl::ROIWithVariablePoints::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f), pt3(10.0f, 32.0f), pt4(21.0f, 25.0f), pt5(28.0f, 3.5f);
		const swl::ROIWithVariablePoints::region_type limitRegion(swl::ROIWithVariablePoints::point_type(-5.0f, -5.0f), swl::ROIWithVariablePoints::point_type(50.0f, 50.0f));

		boost::scoped_ptr<swl::ROIWithVariablePoints> roi(new swl::PolylineROI(true, swl::PolylineROI::real_type(1), swl::PolylineROI::real_type(1), swl::PolylineROI::color_type(), swl::PolylineROI::color_type()));
		BOOST_CHECK(roi);

		{
			roi->clearAllPoints();
			roi->addPoint(pt1);
			roi->addPoint(pt2);
			roi->addPoint(pt3);
			roi->addPoint(pt4);
			roi->addPoint(pt5);

			const swl::ROIWithVariablePoints::point_type delta(5.0f, 10.0f);

			roi->moveRegion(delta, limitRegion);
			BOOST_CHECK(local::comparePoints(roi->getPoint(0), pt1 + delta));
			BOOST_CHECK(local::comparePoints(roi->getPoint(1), pt2 + delta));
		}

		{
			roi->clearAllPoints();
			roi->addPoint(pt1);
			roi->addPoint(pt2);
			roi->addPoint(pt3);
			roi->addPoint(pt4);
			roi->addPoint(pt5);

			const swl::ROIWithVariablePoints::point_type bigDelta(100.0f, -100.0f);
			const swl::ROIWithVariablePoints::real_type dx = 10.0f, dy = -8.5f;  // actual displacement

			roi->moveRegion(bigDelta, limitRegion);
			BOOST_CHECK(local::comparePoints(roi->getPoint(0), pt1 + swl::ROIWithVariablePoints::point_type(dx, dy)));  // caution: not (-5, 1.5), but (-10, 1.5)
			BOOST_CHECK(local::comparePoints(roi->getPoint(1), pt2 + swl::ROIWithVariablePoints::point_type(dx, dy)));
		}

		{
			roi->clearAllPoints();
			roi->addPoint(pt1);
			roi->addPoint(pt2);
			roi->addPoint(pt3);
			roi->addPoint(pt4);
			roi->addPoint(pt5);

			const swl::ROIWithVariablePoints::point_type delta(-5.0f, 100.0f);
			const swl::ROIWithVariablePoints::real_type dx = -5.0f, dy = 18.0f;  // computed displacement
			const swl::ROIWithVariablePoints::real_type dx2 = 0.0f;  // actual displacement

			roi->moveRegion(delta, limitRegion);
			BOOST_CHECK(!local::comparePoints(roi->getPoint(0), pt1 + swl::ROIWithVariablePoints::point_type(dx, dy)));  // caution: not (-25, 28), but (-20, 28)  ==>  don't move along x-axis because x-value is beyond a limit region & away from its boundary
			BOOST_CHECK(local::comparePoints(roi->getPoint(0), pt1 + swl::ROIWithVariablePoints::point_type(dx2, dy)));
			BOOST_CHECK(!local::comparePoints(roi->getPoint(1), pt2 + swl::ROIWithVariablePoints::point_type(dx, dy)));
			BOOST_CHECK(local::comparePoints(roi->getPoint(1), pt2 + swl::ROIWithVariablePoints::point_type(dx2, dy)));
		}
	}

	void testIsVertex()
	{
		Fixture fixture;

		const swl::ROIWithVariablePoints::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f), pt3(10.0f, 30.0f), pt4(21.0f, 25.0f), pt5(28.0f, 3.5f);

		boost::scoped_ptr<swl::ROIWithVariablePoints> roi(new swl::PolylineROI(true, swl::PolylineROI::real_type(1), swl::PolylineROI::real_type(1), swl::PolylineROI::color_type(), swl::PolylineROI::color_type()));
		BOOST_CHECK(roi);

		roi->addPoint(pt1);
		roi->addPoint(pt2);
		roi->addPoint(pt3);
		roi->addPoint(pt4);
		roi->addPoint(pt5);

		BOOST_CHECK(roi->isVertex(swl::RectangleROI::point_type(-20.1f, 10.0f), swl::RectangleROI::real_type(0.5)));
		BOOST_CHECK(!roi->isVertex(swl::RectangleROI::point_type(-20.1f, 10.0f), swl::RectangleROI::real_type(0.05)));
		BOOST_CHECK(roi->isVertex(swl::RectangleROI::point_type(41.0f, 23.5f), swl::RectangleROI::real_type(3)));
		BOOST_CHECK(!roi->isVertex(swl::RectangleROI::point_type(40.1f, 25.3f), swl::RectangleROI::real_type(0.2)));

		BOOST_CHECK(!roi->isVertex((pt1 + pt2) / swl::RectangleROI::real_type(2), swl::RectangleROI::real_type(0.01)));
		BOOST_CHECK(!roi->isVertex((pt2 + pt3) / swl::RectangleROI::real_type(2), swl::RectangleROI::real_type(0.01)));
		BOOST_CHECK(!roi->isVertex((pt3 + pt4) / swl::RectangleROI::real_type(2), swl::RectangleROI::real_type(0.01)));
		BOOST_CHECK(!roi->isVertex((pt4 + pt5) / swl::RectangleROI::real_type(2), swl::RectangleROI::real_type(0.01)));
		BOOST_CHECK(!roi->isVertex((pt5 + pt1) / swl::RectangleROI::real_type(2), swl::RectangleROI::real_type(0.01)));
	}
};

struct ROIWithVariablePointsTestSuite: public boost::unit_test_framework::test_suite
{
	ROIWithVariablePointsTestSuite()
	: boost::unit_test_framework::test_suite("SWL.Util.ROIWithVariablePoints")
	{
		boost::shared_ptr<ROIWithVariablePointsTest> test(new ROIWithVariablePointsTest());

		add(BOOST_CLASS_TEST_CASE(&ROIWithVariablePointsTest::testHandlePoint, test), 0);
		add(BOOST_CLASS_TEST_CASE(&ROIWithVariablePointsTest::testMoveVertex, test), 0);
		add(BOOST_CLASS_TEST_CASE(&ROIWithVariablePointsTest::testMoveVertexWithLimit, test), 0);
		add(BOOST_CLASS_TEST_CASE(&ROIWithVariablePointsTest::testMoveRegion, test), 0);
		add(BOOST_CLASS_TEST_CASE(&ROIWithVariablePointsTest::testMoveRegionWithLimit, test), 0);
		add(BOOST_CLASS_TEST_CASE(&ROIWithVariablePointsTest::testIsVertex, test), 0);

		boost::unit_test::framework::master_test_suite().add(this);
	}
} testsuite;

}  // unnamed namespace

//-----------------------------------------------------------------------------
// Google Test

#elif defined(__SWL_UNIT_TEST__USE_GOOGLE_TEST)

class ROIWithVariablePointsTest : public testing::Test
{
protected:
	/*virtual*/ void SetUp()
	{
	}

	/*virtual*/ void TearDown()
	{
	}
};

TEST_F(ROIWithVariablePointsTest, testHandlePoint)
{
	boost::scoped_ptr<swl::ROIWithVariablePoints> roi(new swl::PolylineROI(true, swl::PolylineROI::real_type(1), swl::PolylineROI::real_type(1), swl::PolylineROI::color_type(), swl::PolylineROI::color_type()));
	EXPECT_TRUE(roi);

	EXPECT_FALSE(roi->containPoint());

	const swl::ROIWithVariablePoints::point_type pt1((swl::ROIWithVariablePoints::real_type)std::rand(), (swl::ROIWithVariablePoints::real_type)std::rand());
	roi->addPoint(pt1);
	const swl::ROIWithVariablePoints::point_type pt2((swl::ROIWithVariablePoints::real_type)std::rand(), (swl::ROIWithVariablePoints::real_type)std::rand());
	roi->addPoint(pt2);
	const swl::ROIWithVariablePoints::point_type pt3((swl::ROIWithVariablePoints::real_type)std::rand(), (swl::ROIWithVariablePoints::real_type)std::rand());
	roi->addPoint(pt3);
	const swl::ROIWithVariablePoints::point_type pt4((swl::ROIWithVariablePoints::real_type)std::rand(), (swl::ROIWithVariablePoints::real_type)std::rand());
	roi->addPoint(pt4);
	roi->addPoint(pt4);  // add the same point
	const swl::ROIWithVariablePoints::point_type pt5((swl::ROIWithVariablePoints::real_type)std::rand(), (swl::ROIWithVariablePoints::real_type)std::rand());
	roi->addPoint(pt5);

	EXPECT_TRUE(roi->containPoint());
	EXPECT_EQ((std::size_t)6, roi->countPoint());  // not 5

	EXPECT_TRUE(local::comparePoints(roi->getPoint(0), pt1));
	EXPECT_TRUE(local::comparePoints(roi->getPoint(1), pt2));
	EXPECT_TRUE(local::comparePoints(roi->getPoint(2), pt3));
	EXPECT_TRUE(local::comparePoints(roi->getPoint(3), pt4));
	EXPECT_TRUE(local::comparePoints(roi->getPoint(4), pt4));
	EXPECT_TRUE(local::comparePoints(roi->getPoint(5), pt5));

	EXPECT_THROW(roi->getPoint(6), swl::LogException);

	roi->removePoint(pt4);
	EXPECT_EQ((std::size_t)4, roi->countPoint());  // remove 2 points
	roi->removePoint(pt3);
	EXPECT_EQ((std::size_t)3, roi->countPoint());

	EXPECT_TRUE(local::comparePoints(roi->getPoint(0), pt1));
	EXPECT_TRUE(local::comparePoints(roi->getPoint(1), pt2));
	EXPECT_TRUE(local::comparePoints(roi->getPoint(2), pt5));

	roi->clearAllPoints();
	EXPECT_FALSE(roi->containPoint());
	EXPECT_EQ((std::size_t)0, roi->countPoint());
}

TEST_F(ROIWithVariablePointsTest, testMoveVertex)
{
	const swl::ROIWithVariablePoints::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f), pt3(10.0f, 30.0f), pt4(21.0f, 25.0f), pt5(28.0f, 3.5f);
	const swl::ROIWithVariablePoints::point_type delta(3.0f, -7.0f);

	boost::scoped_ptr<swl::ROIWithVariablePoints> roi(new swl::PolylineROI(true, swl::PolylineROI::real_type(1), swl::PolylineROI::real_type(1), swl::PolylineROI::color_type(), swl::PolylineROI::color_type()));
	EXPECT_TRUE(roi);

	roi->addPoint(pt1);
	roi->addPoint(pt2);
	roi->addPoint(pt3);
	roi->addPoint(pt4);
	roi->addPoint(pt5);

	{
		EXPECT_FALSE(roi->moveVertex(swl::ROIWithVariablePoints::point_type(-21.0f, 10.0f), delta, 0.1f));
		EXPECT_TRUE(local::comparePoints(roi->getPoint(0), pt1));
		EXPECT_TRUE(local::comparePoints(roi->getPoint(1), pt2));

		EXPECT_TRUE(roi->moveVertex(swl::ROIWithVariablePoints::point_type(-21.0f, 10.0f), delta, 2.0f));
		EXPECT_FALSE(local::comparePoints(roi->getPoint(0), pt1));
		EXPECT_TRUE(local::comparePoints(roi->getPoint(0), pt1 + delta));
		EXPECT_TRUE(local::comparePoints(roi->getPoint(1), pt2));
		EXPECT_FALSE(local::comparePoints(roi->getPoint(1), pt2 + delta));
	}

	{
		EXPECT_FALSE(roi->moveVertex(swl::ROIWithVariablePoints::point_type(39.0f, 26.0f), delta, 0.1f));
		EXPECT_TRUE(local::comparePoints(roi->getPoint(0), pt1 + delta));
		EXPECT_TRUE(local::comparePoints(roi->getPoint(1), pt2));

		EXPECT_TRUE(roi->moveVertex(swl::ROIWithVariablePoints::point_type(39.0f, 26.0f), delta, 2.0f));
		EXPECT_TRUE(local::comparePoints(roi->getPoint(0), pt1 + delta));
		EXPECT_FALSE(local::comparePoints(roi->getPoint(1), pt2));
		EXPECT_TRUE(local::comparePoints(roi->getPoint(1), pt2 + delta));
	}
}

TEST_F(ROIWithVariablePointsTest, testMoveVertexWithLimit)
{
	const swl::ROIWithVariablePoints::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f), pt3(10.0f, 30.0f), pt4(21.0f, 25.0f), pt5(28.0f, 3.5f);
	const swl::ROIWithVariablePoints::point_type delta(5.0f, 10.0f);
	const swl::ROIWithVariablePoints::point_type bigDelta(100.0f, -100.0f);
	const swl::ROIWithVariablePoints::region_type limitRegion(swl::ROIWithVariablePoints::point_type(-5.0f, -5.0f), swl::ROIWithVariablePoints::point_type(50.0f, 50.0f));

	boost::scoped_ptr<swl::ROIWithVariablePoints> roi(new swl::PolylineROI(true, swl::PolylineROI::real_type(1), swl::PolylineROI::real_type(1), swl::PolylineROI::color_type(), swl::PolylineROI::color_type()));
	EXPECT_TRUE(roi);

	{
		roi->clearAllPoints();
		roi->addPoint(pt1);
		roi->addPoint(pt2);
		roi->addPoint(pt3);
		roi->addPoint(pt4);
		roi->addPoint(pt5);

		EXPECT_FALSE(roi->moveVertex(swl::ROIWithVariablePoints::point_type(-21.0f, 10.0f), delta, limitRegion, 0.1f));
		EXPECT_TRUE(local::comparePoints(roi->getPoint(0), pt1));
		EXPECT_TRUE(local::comparePoints(roi->getPoint(1), pt2));

		EXPECT_TRUE(roi->moveVertex(swl::ROIWithVariablePoints::point_type(-21.0f, 10.0f), delta, limitRegion, 2.0f));
		EXPECT_TRUE(local::comparePoints(roi->getPoint(0), pt1 + delta));
		EXPECT_TRUE(local::comparePoints(roi->getPoint(1), pt2));
	}

	{
		roi->clearAllPoints();
		roi->addPoint(pt1);
		roi->addPoint(pt2);
		roi->addPoint(pt3);
		roi->addPoint(pt4);
		roi->addPoint(pt5);

		EXPECT_FALSE(roi->moveVertex(swl::ROIWithVariablePoints::point_type(39.0f, 26.0f), delta, limitRegion, 0.1f));
		EXPECT_TRUE(local::comparePoints(roi->getPoint(0), pt1));
		EXPECT_TRUE(local::comparePoints(roi->getPoint(1), pt2));

		EXPECT_TRUE(roi->moveVertex(swl::ROIWithVariablePoints::point_type(39.0f, 26.0f), delta, limitRegion, 2.0f));
		EXPECT_TRUE(local::comparePoints(roi->getPoint(0), pt1));
		EXPECT_TRUE(local::comparePoints(roi->getPoint(1), pt2 + delta));
	}

	{
		roi->clearAllPoints();
		roi->addPoint(pt1);
		roi->addPoint(pt2);
		roi->addPoint(pt3);
		roi->addPoint(pt4);
		roi->addPoint(pt5);

		EXPECT_TRUE(roi->moveVertex(swl::ROIWithVariablePoints::point_type(-21.0f, 10.0f), bigDelta, limitRegion, 2.0f));
		EXPECT_TRUE(local::comparePoints(roi->getPoint(0), swl::ROIWithVariablePoints::point_type(limitRegion.right, limitRegion.bottom)));
		EXPECT_FALSE(local::comparePoints(roi->getPoint(1), swl::ROIWithVariablePoints::point_type(limitRegion.right, limitRegion.bottom)));
		EXPECT_TRUE(local::comparePoints(roi->getPoint(1), pt2));
	}

	{
		roi->clearAllPoints();
		roi->addPoint(pt1);
		roi->addPoint(pt2);
		roi->addPoint(pt3);
		roi->addPoint(pt4);
		roi->addPoint(pt5);

		EXPECT_TRUE(roi->moveVertex(swl::ROIWithVariablePoints::point_type(39.0f, 26.0f), bigDelta, limitRegion, 2.0f));
		EXPECT_FALSE(local::comparePoints(roi->getPoint(0), swl::ROIWithVariablePoints::point_type(limitRegion.right, limitRegion.bottom)));
		EXPECT_TRUE(local::comparePoints(roi->getPoint(0), pt1));
		EXPECT_TRUE(local::comparePoints(roi->getPoint(1), swl::ROIWithVariablePoints::point_type(limitRegion.right, limitRegion.bottom)));
	}
}

TEST_F(ROIWithVariablePointsTest, testMoveRegion)
{
	const swl::ROIWithVariablePoints::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f), pt3(10.0f, 30.0f), pt4(21.0f, 25.0f), pt5(28.0f, 3.5f);
	const swl::ROIWithVariablePoints::point_type delta(3.0f, -7.0f);

	boost::scoped_ptr<swl::ROIWithVariablePoints> roi(new swl::PolylineROI(true, swl::PolylineROI::real_type(1), swl::PolylineROI::real_type(1), swl::PolylineROI::color_type(), swl::PolylineROI::color_type()));
	EXPECT_TRUE(roi);

	roi->addPoint(pt1);
	roi->addPoint(pt2);
	roi->addPoint(pt3);
	roi->addPoint(pt4);
	roi->addPoint(pt5);

	roi->moveRegion(delta);
	EXPECT_TRUE(local::comparePoints(roi->getPoint(0), pt1 + delta));
	EXPECT_TRUE(local::comparePoints(roi->getPoint(1), pt2 + delta));
	EXPECT_TRUE(local::comparePoints(roi->getPoint(2), pt3 + delta));
	EXPECT_TRUE(local::comparePoints(roi->getPoint(3), pt4 + delta));
	EXPECT_TRUE(local::comparePoints(roi->getPoint(4), pt5 + delta));
}

TEST_F(ROIWithVariablePointsTest, testMoveRegionWithLimit)
{
	const swl::ROIWithVariablePoints::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f), pt3(10.0f, 32.0f), pt4(21.0f, 25.0f), pt5(28.0f, 3.5f);
	const swl::ROIWithVariablePoints::region_type limitRegion(swl::ROIWithVariablePoints::point_type(-5.0f, -5.0f), swl::ROIWithVariablePoints::point_type(50.0f, 50.0f));

	boost::scoped_ptr<swl::ROIWithVariablePoints> roi(new swl::PolylineROI(true, swl::PolylineROI::real_type(1), swl::PolylineROI::real_type(1), swl::PolylineROI::color_type(), swl::PolylineROI::color_type()));
	EXPECT_TRUE(roi);

	{
		roi->clearAllPoints();
		roi->addPoint(pt1);
		roi->addPoint(pt2);
		roi->addPoint(pt3);
		roi->addPoint(pt4);
		roi->addPoint(pt5);

		const swl::ROIWithVariablePoints::point_type delta(5.0f, 10.0f);

		roi->moveRegion(delta, limitRegion);
		EXPECT_TRUE(local::comparePoints(roi->getPoint(0), pt1 + delta));
		EXPECT_TRUE(local::comparePoints(roi->getPoint(1), pt2 + delta));
	}

	{
		roi->clearAllPoints();
		roi->addPoint(pt1);
		roi->addPoint(pt2);
		roi->addPoint(pt3);
		roi->addPoint(pt4);
		roi->addPoint(pt5);

		const swl::ROIWithVariablePoints::point_type bigDelta(100.0f, -100.0f);
		const swl::ROIWithVariablePoints::real_type dx = 10.0f, dy = -8.5f;  // actual displacement

		roi->moveRegion(bigDelta, limitRegion);
		EXPECT_TRUE(local::comparePoints(roi->getPoint(0), pt1 + swl::ROIWithVariablePoints::point_type(dx, dy)));  // caution: not (-5, 1.5), but (-10, 1.5)
		EXPECT_TRUE(local::comparePoints(roi->getPoint(1), pt2 + swl::ROIWithVariablePoints::point_type(dx, dy)));
	}

	{
		roi->clearAllPoints();
		roi->addPoint(pt1);
		roi->addPoint(pt2);
		roi->addPoint(pt3);
		roi->addPoint(pt4);
		roi->addPoint(pt5);

		const swl::ROIWithVariablePoints::point_type delta(-5.0f, 100.0f);
		const swl::ROIWithVariablePoints::real_type dx = -5.0f, dy = 18.0f;  // computed displacement
		const swl::ROIWithVariablePoints::real_type dx2 = 0.0f;  // actual displacement

		roi->moveRegion(delta, limitRegion);
		EXPECT_FALSE(local::comparePoints(roi->getPoint(0), pt1 + swl::ROIWithVariablePoints::point_type(dx, dy)));  // caution: not (-25, 28), but (-20, 28)  ==>  don't move along x-axis because x-value is beyond a limit region & away from its boundary
		EXPECT_TRUE(local::comparePoints(roi->getPoint(0), pt1 + swl::ROIWithVariablePoints::point_type(dx2, dy)));
		EXPECT_FALSE(local::comparePoints(roi->getPoint(1), pt2 + swl::ROIWithVariablePoints::point_type(dx, dy)));
		EXPECT_TRUE(local::comparePoints(roi->getPoint(1), pt2 + swl::ROIWithVariablePoints::point_type(dx2, dy)));
	}
}

TEST_F(ROIWithVariablePointsTest, testIsVertex)
{
	const swl::ROIWithVariablePoints::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f), pt3(10.0f, 30.0f), pt4(21.0f, 25.0f), pt5(28.0f, 3.5f);

	boost::scoped_ptr<swl::ROIWithVariablePoints> roi(new swl::PolylineROI(true, swl::PolylineROI::real_type(1), swl::PolylineROI::real_type(1), swl::PolylineROI::color_type(), swl::PolylineROI::color_type()));
	EXPECT_TRUE(roi);

	roi->addPoint(pt1);
	roi->addPoint(pt2);
	roi->addPoint(pt3);
	roi->addPoint(pt4);
	roi->addPoint(pt5);

	EXPECT_TRUE(roi->isVertex(swl::RectangleROI::point_type(-20.1f, 10.0f), swl::RectangleROI::real_type(0.5)));
	EXPECT_FALSE(roi->isVertex(swl::RectangleROI::point_type(-20.1f, 10.0f), swl::RectangleROI::real_type(0.05)));
	EXPECT_TRUE(roi->isVertex(swl::RectangleROI::point_type(41.0f, 23.5f), swl::RectangleROI::real_type(3)));
	EXPECT_FALSE(roi->isVertex(swl::RectangleROI::point_type(40.1f, 25.3f), swl::RectangleROI::real_type(0.2)));

	EXPECT_FALSE(roi->isVertex((pt1 + pt2) / swl::RectangleROI::real_type(2), swl::RectangleROI::real_type(0.01)));
	EXPECT_FALSE(roi->isVertex((pt2 + pt3) / swl::RectangleROI::real_type(2), swl::RectangleROI::real_type(0.01)));
	EXPECT_FALSE(roi->isVertex((pt3 + pt4) / swl::RectangleROI::real_type(2), swl::RectangleROI::real_type(0.01)));
	EXPECT_FALSE(roi->isVertex((pt4 + pt5) / swl::RectangleROI::real_type(2), swl::RectangleROI::real_type(0.01)));
	EXPECT_FALSE(roi->isVertex((pt5 + pt1) / swl::RectangleROI::real_type(2), swl::RectangleROI::real_type(0.01)));
}

//-----------------------------------------------------------------------------
// CppUnit

#elif defined(__SWL_UNIT_TEST__USE_CPP_UNIT)

struct ROIWithVariablePointsTest: public CppUnit::TestFixture
{
private:
	CPPUNIT_TEST_SUITE(ROIWithVariablePointsTest);
	CPPUNIT_TEST(testHandlePoint);
	CPPUNIT_TEST(testMoveVertex);
	CPPUNIT_TEST(testMoveVertexWithLimit);
	CPPUNIT_TEST(testMoveRegion);
	CPPUNIT_TEST(testMoveRegionWithLimit);
	CPPUNIT_TEST(testIsVertex);
	CPPUNIT_TEST_SUITE_END();

public:
	void setUp()  // set up
	{
		std::srand((unsigned int)std::time(NULL));
	}

	void tearDown()  // tear down
	{
	}

	void testHandlePoint()
	{
		boost::scoped_ptr<swl::ROIWithVariablePoints> roi(new swl::PolylineROI(true, swl::PolylineROI::real_type(1), swl::PolylineROI::real_type(1), swl::PolylineROI::color_type(), swl::PolylineROI::color_type()));
		CPPUNIT_ASSERT(roi);

		CPPUNIT_ASSERT(!roi->containPoint());

		const swl::ROIWithVariablePoints::point_type pt1((swl::ROIWithVariablePoints::real_type)std::rand(), (swl::ROIWithVariablePoints::real_type)std::rand());
		roi->addPoint(pt1);
		const swl::ROIWithVariablePoints::point_type pt2((swl::ROIWithVariablePoints::real_type)std::rand(), (swl::ROIWithVariablePoints::real_type)std::rand());
		roi->addPoint(pt2);
		const swl::ROIWithVariablePoints::point_type pt3((swl::ROIWithVariablePoints::real_type)std::rand(), (swl::ROIWithVariablePoints::real_type)std::rand());
		roi->addPoint(pt3);
		const swl::ROIWithVariablePoints::point_type pt4((swl::ROIWithVariablePoints::real_type)std::rand(), (swl::ROIWithVariablePoints::real_type)std::rand());
		roi->addPoint(pt4);
		roi->addPoint(pt4);  // add the same point
		const swl::ROIWithVariablePoints::point_type pt5((swl::ROIWithVariablePoints::real_type)std::rand(), (swl::ROIWithVariablePoints::real_type)std::rand());
		roi->addPoint(pt5);

		CPPUNIT_ASSERT(roi->containPoint());
		CPPUNIT_ASSERT_EQUAL((std::size_t)6, roi->countPoint());  // not 5

		CPPUNIT_ASSERT(local::comparePoints(roi->getPoint(0), pt1));
		CPPUNIT_ASSERT(local::comparePoints(roi->getPoint(1), pt2));
		CPPUNIT_ASSERT(local::comparePoints(roi->getPoint(2), pt3));
		CPPUNIT_ASSERT(local::comparePoints(roi->getPoint(3), pt4));
		CPPUNIT_ASSERT(local::comparePoints(roi->getPoint(4), pt4));
		CPPUNIT_ASSERT(local::comparePoints(roi->getPoint(5), pt5));

		CPPUNIT_ASSERT_THROW(roi->getPoint(6), swl::LogException);

		roi->removePoint(pt4);
		CPPUNIT_ASSERT_EQUAL((std::size_t)4, roi->countPoint());  // remove 2 points
		roi->removePoint(pt3);
		CPPUNIT_ASSERT_EQUAL((std::size_t)3, roi->countPoint());

		CPPUNIT_ASSERT(local::comparePoints(roi->getPoint(0), pt1));
		CPPUNIT_ASSERT(local::comparePoints(roi->getPoint(1), pt2));
		CPPUNIT_ASSERT(local::comparePoints(roi->getPoint(2), pt5));

		roi->clearAllPoints();
		CPPUNIT_ASSERT(!roi->containPoint());
		CPPUNIT_ASSERT_EQUAL((std::size_t)0, roi->countPoint());
	}

	void testMoveVertex()
	{
		const swl::ROIWithVariablePoints::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f), pt3(10.0f, 30.0f), pt4(21.0f, 25.0f), pt5(28.0f, 3.5f);
		const swl::ROIWithVariablePoints::point_type delta(3.0f, -7.0f);

		boost::scoped_ptr<swl::ROIWithVariablePoints> roi(new swl::PolylineROI(true, swl::PolylineROI::real_type(1), swl::PolylineROI::real_type(1), swl::PolylineROI::color_type(), swl::PolylineROI::color_type()));
		CPPUNIT_ASSERT(roi);

		roi->addPoint(pt1);
		roi->addPoint(pt2);
		roi->addPoint(pt3);
		roi->addPoint(pt4);
		roi->addPoint(pt5);

		{
			CPPUNIT_ASSERT(!roi->moveVertex(swl::ROIWithVariablePoints::point_type(-21.0f, 10.0f), delta, 0.1f));
			CPPUNIT_ASSERT(local::comparePoints(roi->getPoint(0), pt1));
			CPPUNIT_ASSERT(local::comparePoints(roi->getPoint(1), pt2));

			CPPUNIT_ASSERT(roi->moveVertex(swl::ROIWithVariablePoints::point_type(-21.0f, 10.0f), delta, 2.0f));
			CPPUNIT_ASSERT(!local::comparePoints(roi->getPoint(0), pt1));
			CPPUNIT_ASSERT(local::comparePoints(roi->getPoint(0), pt1 + delta));
			CPPUNIT_ASSERT(local::comparePoints(roi->getPoint(1), pt2));
			CPPUNIT_ASSERT(!local::comparePoints(roi->getPoint(1), pt2 + delta));
		}

		{
			CPPUNIT_ASSERT(!roi->moveVertex(swl::ROIWithVariablePoints::point_type(39.0f, 26.0f), delta, 0.1f));
			CPPUNIT_ASSERT(local::comparePoints(roi->getPoint(0), pt1 + delta));
			CPPUNIT_ASSERT(local::comparePoints(roi->getPoint(1), pt2));

			CPPUNIT_ASSERT(roi->moveVertex(swl::ROIWithVariablePoints::point_type(39.0f, 26.0f), delta, 2.0f));
			CPPUNIT_ASSERT(local::comparePoints(roi->getPoint(0), pt1 + delta));
			CPPUNIT_ASSERT(!local::comparePoints(roi->getPoint(1), pt2));
			CPPUNIT_ASSERT(local::comparePoints(roi->getPoint(1), pt2 + delta));
		}
	}

	void testMoveVertexWithLimit()
	{
		const swl::ROIWithVariablePoints::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f), pt3(10.0f, 30.0f), pt4(21.0f, 25.0f), pt5(28.0f, 3.5f);
		const swl::ROIWithVariablePoints::point_type delta(5.0f, 10.0f);
		const swl::ROIWithVariablePoints::point_type bigDelta(100.0f, -100.0f);
		const swl::ROIWithVariablePoints::region_type limitRegion(swl::ROIWithVariablePoints::point_type(-5.0f, -5.0f), swl::ROIWithVariablePoints::point_type(50.0f, 50.0f));

		boost::scoped_ptr<swl::ROIWithVariablePoints> roi(new swl::PolylineROI(true, swl::PolylineROI::real_type(1), swl::PolylineROI::real_type(1), swl::PolylineROI::color_type(), swl::PolylineROI::color_type()));
		CPPUNIT_ASSERT(roi);

		{
			roi->clearAllPoints();
			roi->addPoint(pt1);
			roi->addPoint(pt2);
			roi->addPoint(pt3);
			roi->addPoint(pt4);
			roi->addPoint(pt5);

			CPPUNIT_ASSERT(!roi->moveVertex(swl::ROIWithVariablePoints::point_type(-21.0f, 10.0f), delta, limitRegion, 0.1f));
			CPPUNIT_ASSERT(local::comparePoints(roi->getPoint(0), pt1));
			CPPUNIT_ASSERT(local::comparePoints(roi->getPoint(1), pt2));

			CPPUNIT_ASSERT(roi->moveVertex(swl::ROIWithVariablePoints::point_type(-21.0f, 10.0f), delta, limitRegion, 2.0f));
			CPPUNIT_ASSERT(local::comparePoints(roi->getPoint(0), pt1 + delta));
			CPPUNIT_ASSERT(local::comparePoints(roi->getPoint(1), pt2));
		}

		{
			roi->clearAllPoints();
			roi->addPoint(pt1);
			roi->addPoint(pt2);
			roi->addPoint(pt3);
			roi->addPoint(pt4);
			roi->addPoint(pt5);

			CPPUNIT_ASSERT(!roi->moveVertex(swl::ROIWithVariablePoints::point_type(39.0f, 26.0f), delta, limitRegion, 0.1f));
			CPPUNIT_ASSERT(local::comparePoints(roi->getPoint(0), pt1));
			CPPUNIT_ASSERT(local::comparePoints(roi->getPoint(1), pt2));

			CPPUNIT_ASSERT(roi->moveVertex(swl::ROIWithVariablePoints::point_type(39.0f, 26.0f), delta, limitRegion, 2.0f));
			CPPUNIT_ASSERT(local::comparePoints(roi->getPoint(0), pt1));
			CPPUNIT_ASSERT(local::comparePoints(roi->getPoint(1), pt2 + delta));
		}

		{
			roi->clearAllPoints();
			roi->addPoint(pt1);
			roi->addPoint(pt2);
			roi->addPoint(pt3);
			roi->addPoint(pt4);
			roi->addPoint(pt5);

			CPPUNIT_ASSERT(roi->moveVertex(swl::ROIWithVariablePoints::point_type(-21.0f, 10.0f), bigDelta, limitRegion, 2.0f));
			CPPUNIT_ASSERT(local::comparePoints(roi->getPoint(0), swl::ROIWithVariablePoints::point_type(limitRegion.right, limitRegion.bottom)));
			CPPUNIT_ASSERT(!local::comparePoints(roi->getPoint(1), swl::ROIWithVariablePoints::point_type(limitRegion.right, limitRegion.bottom)));
			CPPUNIT_ASSERT(local::comparePoints(roi->getPoint(1), pt2));
		}

		{
			roi->clearAllPoints();
			roi->addPoint(pt1);
			roi->addPoint(pt2);
			roi->addPoint(pt3);
			roi->addPoint(pt4);
			roi->addPoint(pt5);

			CPPUNIT_ASSERT(roi->moveVertex(swl::ROIWithVariablePoints::point_type(39.0f, 26.0f), bigDelta, limitRegion, 2.0f));
			CPPUNIT_ASSERT(!local::comparePoints(roi->getPoint(0), swl::ROIWithVariablePoints::point_type(limitRegion.right, limitRegion.bottom)));
			CPPUNIT_ASSERT(local::comparePoints(roi->getPoint(0), pt1));
			CPPUNIT_ASSERT(local::comparePoints(roi->getPoint(1), swl::ROIWithVariablePoints::point_type(limitRegion.right, limitRegion.bottom)));
		}
	}

	void testMoveRegion()
	{
		const swl::ROIWithVariablePoints::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f), pt3(10.0f, 30.0f), pt4(21.0f, 25.0f), pt5(28.0f, 3.5f);
		const swl::ROIWithVariablePoints::point_type delta(3.0f, -7.0f);

		boost::scoped_ptr<swl::ROIWithVariablePoints> roi(new swl::PolylineROI(true, swl::PolylineROI::real_type(1), swl::PolylineROI::real_type(1), swl::PolylineROI::color_type(), swl::PolylineROI::color_type()));
		CPPUNIT_ASSERT(roi);

		roi->addPoint(pt1);
		roi->addPoint(pt2);
		roi->addPoint(pt3);
		roi->addPoint(pt4);
		roi->addPoint(pt5);

		roi->moveRegion(delta);
		CPPUNIT_ASSERT(local::comparePoints(roi->getPoint(0), pt1 + delta));
		CPPUNIT_ASSERT(local::comparePoints(roi->getPoint(1), pt2 + delta));
		CPPUNIT_ASSERT(local::comparePoints(roi->getPoint(2), pt3 + delta));
		CPPUNIT_ASSERT(local::comparePoints(roi->getPoint(3), pt4 + delta));
		CPPUNIT_ASSERT(local::comparePoints(roi->getPoint(4), pt5 + delta));
	}

	void testMoveRegionWithLimit()
	{
		const swl::ROIWithVariablePoints::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f), pt3(10.0f, 32.0f), pt4(21.0f, 25.0f), pt5(28.0f, 3.5f);
		const swl::ROIWithVariablePoints::region_type limitRegion(swl::ROIWithVariablePoints::point_type(-5.0f, -5.0f), swl::ROIWithVariablePoints::point_type(50.0f, 50.0f));

		boost::scoped_ptr<swl::ROIWithVariablePoints> roi(new swl::PolylineROI(true, swl::PolylineROI::real_type(1), swl::PolylineROI::real_type(1), swl::PolylineROI::color_type(), swl::PolylineROI::color_type()));
		CPPUNIT_ASSERT(roi);

		{
			roi->clearAllPoints();
			roi->addPoint(pt1);
			roi->addPoint(pt2);
			roi->addPoint(pt3);
			roi->addPoint(pt4);
			roi->addPoint(pt5);

			const swl::ROIWithVariablePoints::point_type delta(5.0f, 10.0f);

			roi->moveRegion(delta, limitRegion);
			CPPUNIT_ASSERT(local::comparePoints(roi->getPoint(0), pt1 + delta));
			CPPUNIT_ASSERT(local::comparePoints(roi->getPoint(1), pt2 + delta));
		}

		{
			roi->clearAllPoints();
			roi->addPoint(pt1);
			roi->addPoint(pt2);
			roi->addPoint(pt3);
			roi->addPoint(pt4);
			roi->addPoint(pt5);

			const swl::ROIWithVariablePoints::point_type bigDelta(100.0f, -100.0f);
			const swl::ROIWithVariablePoints::real_type dx = 10.0f, dy = -8.5f;  // actual displacement

			roi->moveRegion(bigDelta, limitRegion);
			CPPUNIT_ASSERT(local::comparePoints(roi->getPoint(0), pt1 + swl::ROIWithVariablePoints::point_type(dx, dy)));  // caution: not (-5, 1.5), but (-10, 1.5)
			CPPUNIT_ASSERT(local::comparePoints(roi->getPoint(1), pt2 + swl::ROIWithVariablePoints::point_type(dx, dy)));
		}

		{
			roi->clearAllPoints();
			roi->addPoint(pt1);
			roi->addPoint(pt2);
			roi->addPoint(pt3);
			roi->addPoint(pt4);
			roi->addPoint(pt5);

			const swl::ROIWithVariablePoints::point_type delta(-5.0f, 100.0f);
			const swl::ROIWithVariablePoints::real_type dx = -5.0f, dy = 18.0f;  // computed displacement
			const swl::ROIWithVariablePoints::real_type dx2 = 0.0f;  // actual displacement

			roi->moveRegion(delta, limitRegion);
			CPPUNIT_ASSERT(!local::comparePoints(roi->getPoint(0), pt1 + swl::ROIWithVariablePoints::point_type(dx, dy)));  // caution: not (-25, 28), but (-20, 28)  ==>  don't move along x-axis because x-value is beyond a limit region & away from its boundary
			CPPUNIT_ASSERT(local::comparePoints(roi->getPoint(0), pt1 + swl::ROIWithVariablePoints::point_type(dx2, dy)));
			CPPUNIT_ASSERT(!local::comparePoints(roi->getPoint(1), pt2 + swl::ROIWithVariablePoints::point_type(dx, dy)));
			CPPUNIT_ASSERT(local::comparePoints(roi->getPoint(1), pt2 + swl::ROIWithVariablePoints::point_type(dx2, dy)));
		}
	}

	void testIsVertex()
	{
		const swl::ROIWithVariablePoints::point_type pt1(-20.0f, 10.0f), pt2(40.0f, 25.0f), pt3(10.0f, 30.0f), pt4(21.0f, 25.0f), pt5(28.0f, 3.5f);

		boost::scoped_ptr<swl::ROIWithVariablePoints> roi(new swl::PolylineROI(true, swl::PolylineROI::real_type(1), swl::PolylineROI::real_type(1), swl::PolylineROI::color_type(), swl::PolylineROI::color_type()));
		CPPUNIT_ASSERT(roi);

		roi->addPoint(pt1);
		roi->addPoint(pt2);
		roi->addPoint(pt3);
		roi->addPoint(pt4);
		roi->addPoint(pt5);

		CPPUNIT_ASSERT(roi->isVertex(swl::RectangleROI::point_type(-20.1f, 10.0f), swl::RectangleROI::real_type(0.5)));
		CPPUNIT_ASSERT(!roi->isVertex(swl::RectangleROI::point_type(-20.1f, 10.0f), swl::RectangleROI::real_type(0.05)));
		CPPUNIT_ASSERT(roi->isVertex(swl::RectangleROI::point_type(41.0f, 23.5f), swl::RectangleROI::real_type(3)));
		CPPUNIT_ASSERT(!roi->isVertex(swl::RectangleROI::point_type(40.1f, 25.3f), swl::RectangleROI::real_type(0.2)));

		CPPUNIT_ASSERT(!roi->isVertex((pt1 + pt2) / swl::RectangleROI::real_type(2), swl::RectangleROI::real_type(0.01)));
		CPPUNIT_ASSERT(!roi->isVertex((pt2 + pt3) / swl::RectangleROI::real_type(2), swl::RectangleROI::real_type(0.01)));
		CPPUNIT_ASSERT(!roi->isVertex((pt3 + pt4) / swl::RectangleROI::real_type(2), swl::RectangleROI::real_type(0.01)));
		CPPUNIT_ASSERT(!roi->isVertex((pt4 + pt5) / swl::RectangleROI::real_type(2), swl::RectangleROI::real_type(0.01)));
		CPPUNIT_ASSERT(!roi->isVertex((pt5 + pt1) / swl::RectangleROI::real_type(2), swl::RectangleROI::real_type(0.01)));
	}
};

#endif

}  // namespace unit_test
}  // namespace swl

#if defined(__SWL_UNIT_TEST__USE_CPP_UNIT)
//CPPUNIT_TEST_SUITE_REGISTRATION(swl::unit_test::ROIWithVariablePointsTest);
CPPUNIT_REGISTRY_ADD_TO_DEFAULT("SWL.Util");
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(swl::unit_test::ROIWithVariablePointsTest, "SWL.Util");
#endif
