#include "swl/Config.h"
#include "../../UnitTestConfig.h"
#include "swl/math/GeometryUtil.h"
#include "swl/base/String.h"
#include <cmath>
#include <random>
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

struct GeometryUtilTest
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
	void testComputeNearestPointWithLine()
	{
		Fixture fixture;

		const size_t NUM_TESTS = 1000;
		const double tol = 1.0e-5;
		const double eps = 1.0e-20;

		std::random_device seedDevice;
		std::mt19937 RNG = std::mt19937(seedDevice());

		std::uniform_real_distribution<double> paramDist(-100000.0, 100000.0);
		std::uniform_real_distribution<double> pointDist(-100000.0, 100000.0);
		for (size_t loop = 0; loop < NUM_TESTS; )
		{
			//const double a = 1.0, b = -1.0, c = 1.0;
			//const double a = 1.0, b = 0.0, c = -1.0;
			//const double a = 0.0, b = 1.0, c = -1.0;
			const double a = paramDist(RNG), b = paramDist(RNG), c = paramDist(RNG);
			if (std::abs(a) < eps && std::abs(b) < eps) continue;

			//const double x0 = 1.0, y0 = -1.0;
			const double x0 = pointDist(RNG), y0 = pointDist(RNG);
			double nearestX, nearestY;
			BOOST_CHECK(GeometryUtil::computeNearestPointWithLine(x0, y0, a, b, c, nearestX, nearestY));
			{
				const double dist1 = std::sqrt((nearestX - x0)*(nearestX - x0) + (nearestY - y0)*(nearestY - y0));
				const double dist2 = std::abs(a * x0 + b * y0 + c) / std::sqrt(a*a + b*b);

				BOOST_CHECK_CLOSE(dist1, dist2, tol);
			}

			++loop;
		}
	}

	void testComputeNearestPointWithPlane()
	{
		Fixture fixture;

		const size_t NUM_TESTS = 1000;
		const double tol = 1.0e-5;
		const double eps = 1.0e-20;

		std::random_device seedDevice;
		std::mt19937 RNG = std::mt19937(seedDevice());

		std::uniform_real_distribution<double> paramDist(-100000.0, 100000.0);
		std::uniform_real_distribution<double> pointDist(-100000.0, 100000.0);
		for (size_t loop = 0; loop < NUM_TESTS; )
		{
			const double a = paramDist(RNG), b = paramDist(RNG), c = paramDist(RNG), d = paramDist(RNG);
			if (std::abs(a) < eps && std::abs(b) < eps && std::abs(c) < eps) continue;

			const double x0 = pointDist(RNG), y0 = pointDist(RNG), z0 = pointDist(RNG);
			double nearestX, nearestY, nearestZ;
			BOOST_CHECK(GeometryUtil::computeNearestPointWithPlane(x0, y0, z0, a, b, c, d, nearestX, nearestY, nearestZ));
			{
				const double dist1 = std::sqrt((nearestX - x0)*(nearestX - x0) + (nearestY - y0)*(nearestY - y0) + (nearestZ - z0)*(nearestZ - z0));
				const double dist2 = std::abs(a * x0 + b * y0 + c * z0 + d) / std::sqrt(a*a + b*b + c*c);

				BOOST_CHECK_CLOSE(dist1, dist2, tol);
			}

			++loop;
		}
	}
};

struct GeometryUtilTestSuite : public boost::unit_test_framework::test_suite
{
	GeometryUtilTestSuite()
	: boost::unit_test_framework::test_suite("SWL.Math.GeometryUtil")
	{
		boost::shared_ptr<GeometryUtilTest> test(new GeometryUtilTest());

		add(BOOST_CLASS_TEST_CASE(&GeometryUtilTest::testComputeNearestPointWithLine, test), 0);
		add(BOOST_CLASS_TEST_CASE(&GeometryUtilTest::testComputeNearestPointWithPlane, test), 0);

		boost::unit_test::framework::master_test_suite().add(this);
	}
} testsuite;

}  // unnamed namespace

//-----------------------------------------------------------------------------
// Google Test.

#elif defined(__SWL_UNIT_TEST__USE_GOOGLE_TEST)

class GeometryUtilTest : public testing::Test
{
protected:
	/*virtual*/ void SetUp()
	{
	}

	/*virtual*/ void TearDown()
	{
	}
};

TEST_F(GeometryUtilTest, testComputeNearestPointWithLine)
{
	const size_t NUM_TESTS = 1000;
	const double tol = 1.0e-5;
	const double eps = 1.0e-20;

	std::random_device seedDevice;
	std::mt19937 RNG = std::mt19937(seedDevice());

	std::uniform_real_distribution<double> paramDist(-100000.0, 100000.0);
	std::uniform_real_distribution<double> pointDist(-100000.0, 100000.0);
	for (size_t loop = 0; loop < NUM_TESTS; )
	{
		//const double a = 1.0, b = -1.0, c = 1.0;
		//const double a = 1.0, b = 0.0, c = -1.0;
		//const double a = 0.0, b = 1.0, c = -1.0;
		const double a = paramDist(RNG), b = paramDist(RNG), c = paramDist(RNG);
		if (std::abs(a) < eps && std::abs(b) < eps) continue;

		//const double x0 = 1.0, y0 = -1.0;
		const double x0 = pointDist(RNG), y0 = pointDist(RNG);
		double nearestX, nearestY;
		EXPECT_TRUE(GeometryUtil::computeNearestPointWithLine(x0, y0, a, b, c, nearestX, nearestY));
		{
			const double dist1 = std::sqrt((nearestX - x0)*(nearestX - x0) + (nearestY - y0)*(nearestY - y0));
			const double dist2 = std::abs(a * x0 + b * y0 + c) / std::sqrt(a*a + b*b);

			EXPECT_NEAR(dist1, dist2, tol);
		}

		++loop;
	}
}

TEST_F(GeometryUtilTest, testComputeNearestPointWithPlane)
{
	const size_t NUM_TESTS = 1000;
	const double tol = 1.0e-5;
	const double eps = 1.0e-20;

	std::random_device seedDevice;
	std::mt19937 RNG = std::mt19937(seedDevice());

	std::uniform_real_distribution<double> paramDist(-100000.0, 100000.0);
	std::uniform_real_distribution<double> pointDist(-100000.0, 100000.0);
	for (size_t loop = 0; loop < NUM_TESTS; )
	{
		const double a = paramDist(RNG), b = paramDist(RNG), c = paramDist(RNG), d = paramDist(RNG);
		if (std::abs(a) < eps && std::abs(b) < eps && std::abs(c) < eps) continue;

		const double x0 = pointDist(RNG), y0 = pointDist(RNG), z0 = pointDist(RNG);
		double nearestX, nearestY, nearestZ;
		EXPECT_TRUE(GeometryUtil::computeNearestPointWithPlane(x0, y0, z0, a, b, c, d, nearestX, nearestY, nearestZ));
		{
			const double dist1 = std::sqrt((nearestX - x0)*(nearestX - x0) + (nearestY - y0)*(nearestY - y0) + (nearestZ - z0)*(nearestZ - z0));
			const double dist2 = std::abs(a * x0 + b * y0 + c * z0 + d) / std::sqrt(a*a + b*b + c*c);

			EXPECT_NEAR(dist1, dist2, tol);
		}

		++loop;
	}
}

//-----------------------------------------------------------------------------
// CppUnit.

#elif defined(__SWL_UNIT_TEST__USE_CPP_UNIT)

struct GeometryUtilTest : public CppUnit::TestFixture
{
private:
	CPPUNIT_TEST_SUITE(GeometryUtilTest);
	CPPUNIT_TEST(testComputeNearestPointWithLine);
	CPPUNIT_TEST(testComputeNearestPointWithPlane);
	CPPUNIT_TEST_SUITE_END();

public:
	void setUp()  // Set up.
	{
	}

	void tearDown()  // Tear down.
	{
	}

	void testComputeNearestPointWithLine()
	{
		const size_t NUM_TESTS = 1000;
		const double tol = 1.0e-5;
		const double eps = 1.0e-20;

		std::random_device seedDevice;
		std::mt19937 RNG = std::mt19937(seedDevice());

		std::uniform_real_distribution<double> paramDist(-100000.0, 100000.0);
		std::uniform_real_distribution<double> pointDist(-100000.0, 100000.0);
		for (size_t loop = 0; loop < NUM_TESTS; )
		{
			//const double a = 1.0, b = -1.0, c = 1.0;
			//const double a = 1.0, b = 0.0, c = -1.0;
			//const double a = 0.0, b = 1.0, c = -1.0;
			const double a = paramDist(RNG), b = paramDist(RNG), c = paramDist(RNG);
			if (std::abs(a) < eps && std::abs(b) < eps) continue;

			//const double x0 = 1.0, y0 = -1.0;
			const double x0 = pointDist(RNG), y0 = pointDist(RNG);
			double nearestX, nearestY;
			CPPUNIT_ASSERT(GeometryUtil::computeNearestPointWithLine(x0, y0, a, b, c, nearestX, nearestY));
			{
				const double dist1 = std::sqrt((nearestX - x0)*(nearestX - x0) + (nearestY - y0)*(nearestY - y0));
				const double dist2 = std::abs(a * x0 + b * y0 + c) / std::sqrt(a*a + b*b);

				CPPUNIT_ASSERT_DOUBLES_EQUAL(dist1, dist2, tol);
			}

			++loop;
		}
	}

	void testComputeNearestPointWithPlane()
	{
		const size_t NUM_TESTS = 1000;
		const double tol = 1.0e-5;
		const double eps = 1.0e-20;

		std::random_device seedDevice;
		std::mt19937 RNG = std::mt19937(seedDevice());

		std::uniform_real_distribution<double> paramDist(-100000.0, 100000.0);
		std::uniform_real_distribution<double> pointDist(-100000.0, 100000.0);
		for (size_t loop = 0; loop < NUM_TESTS; )
		{
			const double a = paramDist(RNG), b = paramDist(RNG), c = paramDist(RNG), d = paramDist(RNG);
			if (std::abs(a) < eps && std::abs(b) < eps && std::abs(c) < eps) continue;

			const double x0 = pointDist(RNG), y0 = pointDist(RNG), z0 = pointDist(RNG);
			double nearestX, nearestY, nearestZ;
			CPPUNIT_ASSERT(GeometryUtil::computeNearestPointWithPlane(x0, y0, z0, a, b, c, d, nearestX, nearestY, nearestZ));
			{
				const double dist1 = std::sqrt((nearestX - x0)*(nearestX - x0) + (nearestY - y0)*(nearestY - y0) + (nearestZ - z0)*(nearestZ - z0));
				const double dist2 = std::abs(a * x0 + b * y0 + c * z0 + d) / std::sqrt(a*a + b*b + c*c);

				CPPUNIT_ASSERT_DOUBLES_EQUAL(dist1, dist2, tol);
			}

			++loop;
		}
	}
};

#endif

}  // namespace unit_test
}  // namespace swl

#if defined(__SWL_UNIT_TEST__USE_CPP_UNIT)
//CPPUNIT_TEST_SUITE_REGISTRATION(swl::unit_test::GeometryUtilTest);
CPPUNIT_REGISTRY_ADD_TO_DEFAULT("SWL.Math");
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(swl::unit_test::GeometryUtilTest, "SWL.Math");
#endif
