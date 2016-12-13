#include "swl/Config.h"
#include "../../UnitTestConfig.h"
#include "swl/base/String.h"
#include "swl/math/Statistic.h"
#include <iostream>


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

struct StatisticTest
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
	StatisticTest()
	: tol_(1.0e-5)
	{}

public:
	// sample mean of a vector.
	void testSampleMeanOfVector()
	{
		Fixture fixture;

		Eigen::VectorXd data(3);
		data << 1, 2, 3;

		const double mean = data.mean();

		const double truth = 2.0;

		BOOST_CHECK_CLOSE(mean, truth, tol_);

		// display.
		//std::cout << "sample mean = " << mean << std::endl;
		//sample mean = 2
	}

	// sample variance of a vector.
	void testSampleVarianceOfVector()
	{
		Fixture fixture;

		Eigen::VectorXd data(3);
		data << 1, 2, 3;

		const double var(swl::Statistic::sampleVariance(data));

		const double truth = 1.0;

		BOOST_CHECK_CLOSE(var, truth, tol_);

		// display.
		//std::cout << "sample variance = " << var << std::endl;
		//sample variance = 1
	}

	// sample means of a matrix
	void testSampleMeanOfMatrix()
	{
		Fixture fixture;

		Eigen::MatrixXd data(3, 3);
		data << 1, 2, 3,  4, 4, 4,  -1, 0, 3;  // row-major matrix.

		const Eigen::VectorXd mean(data.rowwise().mean());

		Eigen::VectorXd truth(3);
		truth << 2, 4, 0.666667;

		BOOST_CHECK(((mean - truth).array().abs() < Eigen::VectorXd::Constant(3, tol_).array()).all());

		// display.
		//std::cout << "sample means = " << mean.transpose() << std::endl;
		//sample means =        2        4 0.666667
	}

	// row-wise sample variances of a matrix.
	void testSampleVariancesOfMatrix()
	{
		Fixture fixture;

		Eigen::MatrixXd data(3, 3);
		data << 1, 2, 3,  4, 4, 4,  -1, 0, 3;  // row-major matrix.

		const Eigen::VectorXd vars(swl::Statistic::sampleVariance(data));

		Eigen::VectorXd truth(3);
		truth << 1, 0, 4.33333;

		BOOST_CHECK(((vars - truth).array().abs() < Eigen::VectorXd::Constant(3, tol_).array()).all());

		// display.
		//std::cout << "sample variances = " << vars.transpose() << std::endl;
		//sample variances =       1       0 4.33333
	}

	// covariance of a matrix.
	void testSampleCovariancesOfMatrix()
	{
		Fixture fixture;

		Eigen::MatrixXd data(3, 3);
		data << 1, 2, 3,  4, 4, 4,  -1, 0, 3;  // row-major matrix.

		const Eigen::MatrixXd cov(swl::Statistic::sampleCovarianceMatrix(data));

		Eigen::MatrixXd truth(3, 3);
		truth << 1, 0, 2,  0, 0, 0,  2, 0, 4.33333;

		BOOST_CHECK(((cov - truth).array().abs() < Eigen::MatrixXd::Constant(3, 3, tol_).array()).all());

		// display.
		//std::cout << "sample covariance matrix = " << std::endl << cov << std::endl;
		/*
		sample covariance matrix =
			  1       0       2
			  0       0       0
			  2       0 4.33333
		*/
	}

private:
	const double tol_;
};

struct StatisticTestSuite: public boost::unit_test_framework::test_suite
{
	StatisticTestSuite()
	: boost::unit_test_framework::test_suite("SWL.Math.Statistic")
	{
		boost::shared_ptr<StatisticTest> test(new StatisticTest());

		add(BOOST_CLASS_TEST_CASE(&StatisticTest::testSampleMeanOfVector, test), 0);
		add(BOOST_CLASS_TEST_CASE(&StatisticTest::testSampleVarianceOfVector, test), 0);
		add(BOOST_CLASS_TEST_CASE(&StatisticTest::testSampleMeanOfMatrix, test), 0);
		add(BOOST_CLASS_TEST_CASE(&StatisticTest::testSampleVariancesOfMatrix, test), 0);
		add(BOOST_CLASS_TEST_CASE(&StatisticTest::testSampleCovariancesOfMatrix, test), 0);

		boost::unit_test::framework::master_test_suite().add(this);
	}
} testsuite;

}  // unnamed namespace

//-----------------------------------------------------------------------------
//

#elif defined(__SWL_UNIT_TEST__USE_CPP_UNIT)

struct StatisticTest: public CppUnit::TestFixture
{
private:
	CPPUNIT_TEST_SUITE(StatisticTest);
	CPPUNIT_TEST(test);
	CPPUNIT_TEST_SUITE_END();

public:
	void setUp()  // Set up.
	{
	}

	void tearDown()  // Tear down.
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
//CPPUNIT_TEST_SUITE_REGISTRATION(swl::unit_test::StatisticTest);
CPPUNIT_REGISTRY_ADD_TO_DEFAULT("SWL.Math");
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(swl::unit_test::StatisticTest, "SWL.Math");
#endif
