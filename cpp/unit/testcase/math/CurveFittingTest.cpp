#include "swl/Config.h"
#include "../../UnitTestConfig.h"
#include "swl/math/CurveFitting.h"
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

struct CurveFittingTest
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
	void testEstimateLineByLeastSquares()
	{
		Fixture fixture;

		const size_t NUM_TESTS = 100;
		const size_t NUM_POINTS = 1000;
		const double tol = 1.0e-3;
		const double eps = std::numeric_limits<double>::epsilon() * 100;

		std::random_device seedDevice;
		std::mt19937 RNG = std::mt19937(seedDevice());

		std::uniform_real_distribution<double> paramDist(-1000.0, 1000.0);
		std::uniform_real_distribution<double> pointDist(-100000.0, 100000.0);
		std::normal_distribution<double> noiseDist(0.0, 10.0);
		for (size_t loop = 0; loop < NUM_TESTS; )
		{
			const double a = paramDist(RNG), b = paramDist(RNG), c = paramDist(RNG);
			if (std::abs(a) < eps && std::abs(b) < eps) continue;

			std::list<std::array<double, 2> > points;
			for (size_t idx = 0; idx < NUM_POINTS; ++idx)
			{
				const double val = pointDist(RNG);
				if (std::abs(a) > eps)
					//points.push_back({ (b * val + c) / -a + noiseDist(RNG), val + noiseDist(RNG) });
					points.push_back({ (b * val + c) / -a, val });
				else
					//points.push_back({ val + noiseDist(RNG), (a * val + c) / -b + noiseDist(RNG) });
					points.push_back({ val, (a * val + c) / -b });
			}

			double ae = 0.0, be = 0.0, ce = 0.0;
			BOOST_CHECK(CurveFitting::estimateLineByLeastSquares(points, ae, be, ce));

			const double denom = std::sqrt(a*a + b*b + c*c);
			//BOOST_CHECK_CLOSE(a / denom, ae, tol);
			//BOOST_CHECK_CLOSE(b / denom, be, tol);
			//BOOST_CHECK_CLOSE(c / denom, ce, tol);

			++loop;
		}
	}

	void testEstimateQuadraticByLeastSquares()
	{
		Fixture fixture;

		const size_t NUM_TESTS = 100;
		const size_t NUM_POINTS = 1000;
		const double tol = 1.0e-3;
		const double eps = std::numeric_limits<double>::epsilon() * 100;

		std::random_device seedDevice;
		std::mt19937 RNG = std::mt19937(seedDevice());

		std::uniform_real_distribution<double> paramDist(-1000.0, 1000.0);
		std::uniform_real_distribution<double> pointDist(-100000.0, 100000.0);
		std::normal_distribution<double> noiseDist(0.0, 10.0);
		std::uniform_int_distribution<> evenOddDist(0, 1);
		for (size_t loop = 0; loop < NUM_TESTS; )
		{
			const double a = paramDist(RNG), b = paramDist(RNG), c = paramDist(RNG), d = paramDist(RNG);
			if (std::abs(a) < eps && std::abs(b) < eps && std::abs(c) < eps) continue;

			std::list<std::array<double, 2> > points;
			for (size_t idx = 0; idx < NUM_POINTS; ++idx)
			{
				const double val = pointDist(RNG);
				if (std::abs(c) > eps)
					points.push_back({ val + noiseDist(RNG), (a * val * val + b * val + d) / -c + noiseDist(RNG) });
				else if (std::abs(a) > eps)
				{
					const double val2 = std::sqrt(0.25*b*b / a - c*val - d) / a;
					const double xx = -0.5 * b / a + (evenOddDist(RNG) % 2 ? val : -val);
					points.push_back({ xx + noiseDist(RNG), val + noiseDist(RNG) });
				}
				else
					points.push_back({ (c * val + d) / -b + noiseDist(RNG), val + noiseDist(RNG) });
			}

			double ae = 0.0, be = 0.0, ce = 0.0, de = 0.0;
			BOOST_CHECK(CurveFitting::estimateQuadraticByLeastSquares(points, ae, be, ce, de));

			const double denom = std::sqrt(a*a + b*b + c*c + d*d);
			//BOOST_CHECK_CLOSE(a / denom, ae, tol);
			//BOOST_CHECK_CLOSE(b / denom, be, tol);
			//BOOST_CHECK_CLOSE(c / denom, ce, tol);
			//BOOST_CHECK_CLOSE(d / denom, de, tol);

			++loop;
		}
	}
};

struct CurveFittingTestSuite : public boost::unit_test_framework::test_suite
{
	CurveFittingTestSuite()
	: boost::unit_test_framework::test_suite("SWL.Math.CurveFitting")
	{
		boost::shared_ptr<CurveFittingTest> test(new CurveFittingTest());

		add(BOOST_CLASS_TEST_CASE(&CurveFittingTest::testEstimateLineByLeastSquares, test), 0);
		add(BOOST_CLASS_TEST_CASE(&CurveFittingTest::testEstimateQuadraticByLeastSquares, test), 0);

		boost::unit_test::framework::master_test_suite().add(this);
	}
} testsuite;

}  // unnamed namespace

//-----------------------------------------------------------------------------
// Google Test.

#elif defined(__SWL_UNIT_TEST__USE_GOOGLE_TEST)

class CurveFittingTest : public testing::Test
{
protected:
	/*virtual*/ void SetUp()
	{
	}

	/*virtual*/ void TearDown()
	{
	}
};

TEST_F(CurveFittingTest, testEstimateLineByLeastSquares)
{
	const size_t NUM_TESTS = 100;
	const size_t NUM_POINTS = 1000;
	//const double tol = 1.0e-3;
	const double eps = std::numeric_limits<double>::epsilon() * 100;

	std::random_device seedDevice;
	std::mt19937 RNG = std::mt19937(seedDevice());

	std::uniform_real_distribution<double> paramDist(-1000.0, 1000.0);
	std::uniform_real_distribution<double> pointDist(-100000.0, 100000.0);
	std::normal_distribution<double> noiseDist(0.0, 10.0);
	for (size_t loop = 0; loop < NUM_TESTS; )
	{
		const double a = paramDist(RNG), b = paramDist(RNG), c = paramDist(RNG);
		if (std::abs(a) < eps && std::abs(b) < eps) continue;

		std::list<std::array<double, 2> > points;
		for (size_t idx = 0; idx < NUM_POINTS; ++idx)
		{
			const double val = pointDist(RNG);
			if (std::abs(a) > eps)
				//points.push_back({ (b * val + c) / -a + noiseDist(RNG), val + noiseDist(RNG) });
				points.push_back({ (b * val + c) / -a, val });
			else
				//points.push_back({ val + noiseDist(RNG), (a * val + c) / -b + noiseDist(RNG) });
				points.push_back({ val, (a * val + c) / -b });
		}

		double ae = 0.0, be = 0.0, ce = 0.0;
		EXPECT_TRUE(CurveFitting::estimateLineByLeastSquares(points, ae, be, ce));

		//const double denom = std::sqrt(a*a + b*b + c*c);
		//EXPECT_NEAR(a / denom, ae, tol);
		//EXPECT_NEAR(b / denom, be, tol);
		//EXPECT_NEAR(c / denom, ce, tol);

		++loop;
	}
}

TEST_F(CurveFittingTest, testEstimateQuadraticByLeastSquares)
{
	const size_t NUM_TESTS = 100;
	const size_t NUM_POINTS = 1000;
	//const double tol = 1.0e-3;
	const double eps = std::numeric_limits<double>::epsilon() * 100;

	std::random_device seedDevice;
	std::mt19937 RNG = std::mt19937(seedDevice());

	std::uniform_real_distribution<double> paramDist(-1000.0, 1000.0);
	std::uniform_real_distribution<double> pointDist(-100000.0, 100000.0);
	std::normal_distribution<double> noiseDist(0.0, 10.0);
	std::uniform_int_distribution<> evenOddDist(0, 1);
	for (size_t loop = 0; loop < NUM_TESTS; )
	{
		const double a = paramDist(RNG), b = paramDist(RNG), c = paramDist(RNG), d = paramDist(RNG);
		if (std::abs(a) < eps && std::abs(b) < eps && std::abs(c) < eps) continue;

		std::list<std::array<double, 2> > points;
		for (size_t idx = 0; idx < NUM_POINTS; ++idx)
		{
			const double val = pointDist(RNG);
			if (std::abs(c) > eps)
				points.push_back({ val + noiseDist(RNG), (a * val * val + b * val + d) / -c + noiseDist(RNG) });
			else if (std::abs(a) > eps)
			{
				//const double val2 = std::sqrt(0.25*b*b / a - c*val - d) / a;
				const double xx = -0.5 * b / a + (evenOddDist(RNG) % 2 ? val : -val);
				points.push_back({ xx + noiseDist(RNG), val + noiseDist(RNG) });
			}
			else
				points.push_back({ (c * val + d) / -b + noiseDist(RNG), val + noiseDist(RNG) });
		}

		double ae = 0.0, be = 0.0, ce = 0.0, de = 0.0;
		EXPECT_TRUE(CurveFitting::estimateQuadraticByLeastSquares(points, ae, be, ce, de));

		//const double denom = std::sqrt(a*a + b*b + c*c + d*d);
		//EXPECT_NEAR(a / denom, ae, tol);
		//EXPECT_NEAR(b / denom, be, tol);
		//EXPECT_NEAR(c / denom, ce, tol);
		//EXPECT_NEAR(d / denom, de, tol);

		++loop;
	}
}

//-----------------------------------------------------------------------------
// CppUnit.

#elif defined(__SWL_UNIT_TEST__USE_CPP_UNIT)

struct CurveFittingTest : public CppUnit::TestFixture
{
private:
	CPPUNIT_TEST_SUITE(CurveFittingTest);
	CPPUNIT_TEST(testEstimateLineByLeastSquares);
	CPPUNIT_TEST(testEstimateQuadraticByLeastSquares);
	CPPUNIT_TEST_SUITE_END();

public:
	void setUp()  // Set up.
	{
	}

	void tearDown()  // Tear down.
	{
	}

	void testEstimateLineByLeastSquares()
	{
		const size_t NUM_TESTS = 100;
		const size_t NUM_POINTS = 1000;
		//const double tol = 1.0e-3;
		const double eps = std::numeric_limits<double>::epsilon() * 100;

		std::random_device seedDevice;
		std::mt19937 RNG = std::mt19937(seedDevice());

		std::uniform_real_distribution<double> paramDist(-1000.0, 1000.0);
		std::uniform_real_distribution<double> pointDist(-100000.0, 100000.0);
		std::normal_distribution<double> noiseDist(0.0, 10.0);
		for (size_t loop = 0; loop < NUM_TESTS; )
		{
			const double a = paramDist(RNG), b = paramDist(RNG), c = paramDist(RNG);
			if (std::abs(a) < eps && std::abs(b) < eps) continue;

			std::list<std::array<double, 2> > points;
			for (size_t idx = 0; idx < NUM_POINTS; ++idx)
			{
				const double val = pointDist(RNG);
				if (std::abs(a) > eps)
					//points.push_back({ (b * val + c) / -a + noiseDist(RNG), val + noiseDist(RNG) });
					points.push_back({ (b * val + c) / -a, val });
				else
					//points.push_back({ val + noiseDist(RNG), (a * val + c) / -b + noiseDist(RNG) });
					points.push_back({ val, (a * val + c) / -b });
			}

			double ae = 0.0, be = 0.0, ce = 0.0;
			CPPUNIT_ASSERT(CurveFitting::estimateLineByLeastSquares(points, ae, be, ce));

			//const double denom = std::sqrt(a*a + b*b + c*c);
			//CPPUNIT_ASSERT_DOUBLES_EQUAL(a / denom, ae, tol);
			//CPPUNIT_ASSERT_DOUBLES_EQUAL(b / denom, be, tol);
			//CPPUNIT_ASSERT_DOUBLES_EQUAL(c / denom, ce, tol);

			++loop;
		}
	}

	void testEstimateQuadraticByLeastSquares()
	{
		const size_t NUM_TESTS = 100;
		const size_t NUM_POINTS = 1000;
		//const double tol = 1.0e-3;
		const double eps = std::numeric_limits<double>::epsilon() * 100;

		std::random_device seedDevice;
		std::mt19937 RNG = std::mt19937(seedDevice());

		std::uniform_real_distribution<double> paramDist(-1000.0, 1000.0);
		std::uniform_real_distribution<double> pointDist(-100000.0, 100000.0);
		std::normal_distribution<double> noiseDist(0.0, 10.0);
		std::uniform_int_distribution<> evenOddDist(0, 1);
		for (size_t loop = 0; loop < NUM_TESTS; )
		{
			const double a = paramDist(RNG), b = paramDist(RNG), c = paramDist(RNG), d = paramDist(RNG);
			if (std::abs(a) < eps && std::abs(b) < eps && std::abs(c) < eps) continue;

			std::list<std::array<double, 2> > points;
			for (size_t idx = 0; idx < NUM_POINTS; ++idx)
			{
				const double val = pointDist(RNG);
				if (std::abs(c) > eps)
					points.push_back({ val + noiseDist(RNG), (a * val * val + b * val + d) / -c + noiseDist(RNG) });
				else if (std::abs(a) > eps)
				{
					//const double val2 = std::sqrt(0.25*b*b / a - c*val - d) / a;
					const double xx = -0.5 * b / a + (evenOddDist(RNG) % 2 ? val : -val);
					points.push_back({ xx + noiseDist(RNG), val + noiseDist(RNG) });
			}
				else
					points.push_back({ (c * val + d) / -b + noiseDist(RNG), val + noiseDist(RNG) });
		}

			double ae = 0.0, be = 0.0, ce = 0.0, de = 0.0;
			CPPUNIT_ASSERT(CurveFitting::estimateQuadraticByLeastSquares(points, ae, be, ce, de));

			//const double denom = std::sqrt(a*a + b*b + c*c + d*d);
			//CPPUNIT_ASSERT_DOUBLES_EQUAL(a / denom, ae, tol);
			//CPPUNIT_ASSERT_DOUBLES_EQUAL(b / denom, be, tol);
			//CPPUNIT_ASSERT_DOUBLES_EQUAL(c / denom, ce, tol);
			//CPPUNIT_ASSERT_DOUBLES_EQUAL(d / denom, de, tol);

			++loop;
	}
	}
};

#endif

}  // namespace unit_test
}  // namespace swl

#if defined(__SWL_UNIT_TEST__USE_CPP_UNIT)
//CPPUNIT_TEST_SUITE_REGISTRATION(swl::unit_test::CurveFittingTest);
CPPUNIT_REGISTRY_ADD_TO_DEFAULT("SWL.Math");
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(swl::unit_test::CurveFittingTest, "SWL.Math");
#endif
