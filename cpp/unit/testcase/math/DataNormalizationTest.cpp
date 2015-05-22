#include "swl/Config.h"
#include "../../UnitTestConfig.h"
#include "swl/base/String.h"
#include "swl/math/DataNormalization.h"
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

struct DataNormalizationTest
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
	DataNormalizationTest()
	: eps_(1.0e-5)
	{}

public:
	// normalize data by centering.
	void testNormalizeDataByCentering()
	{
		Fixture fixture;

		Eigen::MatrixXd data(3, 3);
		data << 1, 2, 3,  4, 4, 4,  -1, 0, 3;  // row-major matrix.

		DataNormalization::normalizeDataByCentering(data);

		Eigen::MatrixXd truth(3, 3);
		truth << -1, 0, 1,  0, 0, 0,  -1.66667, -0.666667, 2.33333;

		BOOST_CHECK(((data - truth).array().abs() < Eigen::MatrixXd::Constant(3, 3, eps_).array()).all());

		// display.
		//std::cout << "data after centering = " << std::endl << data << std::endl;
		//std::cout << "mean = " << data.rowwise().mean().transpose() << std::endl;
		/*
		data after centering =
			   -1         0         1
				0         0         0
		 -1.66667 -0.666667   2.33333
		mean =           0           0 1.4803e-016
		*/
	}

	// normalize data by average distance.
	void testNormalizeDataByAverageDistance()
	{
		Fixture fixture;

		Eigen::MatrixXd data(3, 3);
		data << 1, 2, 3,  4, 4, 4,  -1, 0, 3;  // row-major matrix.

		const double averageDistance = std::sqrt(2.0);
		DataNormalization::normalizeDataByAverageDistance(data, averageDistance);

		Eigen::MatrixXd truth(3, 3);
		truth << 0.377964, 0.755929, 1.13389,  0.816497, 0.816497, 0.816497,  -0.447214, 0, 1.34164;

		BOOST_CHECK(((data - truth).array().abs() < Eigen::MatrixXd::Constant(3, 3, eps_).array()).all());

		// display.
		//std::cout << "data after normalization by average distance = " << std::endl << data << std::endl;
		//std::cout << "distance(2-norm) = " << data.rowwise().norm().transpose() << std::endl;
		/*
		data after normalization by average distance =
		 0.377964  0.755929   1.13389
		 0.816497  0.816497  0.816497
		-0.447214         0   1.34164
		distance(2-norm) = 1.41421 1.41421 1.41421
		*/
	}

	// normalize data by range.
	void testNormalizeDataByRange()
	{
		Fixture fixture;

		Eigen::MatrixXd data(3, 3);
		data << 1, 2, 3,  4, 4, 4,  -1, 0, 3;  // row-major matrix.

		DataNormalization::normalizeDataByRange(data, 0.0, 1.0);

		Eigen::MatrixXd truth(3, 3);
		truth << 0, 0.5, 1,  0.5, 0.5, 0.5,  0, 0.25, 1;

		BOOST_CHECK(((data - truth).array().abs() < Eigen::MatrixXd::Constant(3, 3, eps_).array()).all());

		// display.
		//std::cout << "data after normalization by range = " << std::endl << data << std::endl;
		//std::cout << "mins = " << data.rowwise().minCoeff().transpose() << std::endl;
		//std::cout << "maxs = " << data.rowwise().maxCoeff().transpose() << std::endl;
		/*
		data after normalization by range =
		   0  0.5    1
		 0.5  0.5  0.5
		   0 0.25    1
		mins =   0 0.5   0
		maxs =   1 0.5   1
		*/
	}

	// normalize data by linear transformation.
	void testNormalizeDataByLinearTransformation()
	{
		Fixture fixture;

		Eigen::MatrixXd data(3, 3);
		data << 1, 2, 3,  4, 4, 4,  -1, 0, 3;  // row-major matrix.

		Eigen::MatrixXd T(3, 3);
		T << 1, 0, -1,  -1, 2, -3,  4, -3, 3;  // row-major matrix.

		BOOST_CHECK(DataNormalization::normalizeDataByLinearTransformation(data, T));

		Eigen::MatrixXd truth(3, 3);
		truth << 2, 2, 0,  10, 6, -4,  -11, -4,  9;

		BOOST_CHECK(((data - truth).array().abs() < Eigen::MatrixXd::Constant(3, 3, eps_).array()).all());

		// display.
		//std::cout << "data after normalization by linear transformation = " << std::endl << data << std::endl;
		/*
		data after normalization by linear transformation =
			2   2   0
			10   6  -4
			-11  -4   9
		*/
	}

	// normalize data by homogeneous transformation.
	void testNormalizeDataByHomogeneousTransformation()
	{
		Fixture fixture;

		Eigen::MatrixXd data(3, 3);
		data << 1, 2, 3,  4, 4, 4,  -1, 0, 3;  // row-major matrix.

		Eigen::MatrixXd H(3, 4);
		H << 1, 0, -1, -3,  -1, 2, -3, -2,  4, -3, 3, -1;  // row-major matrix.

		BOOST_CHECK(DataNormalization::normalizeDataByHomogeneousTransformation(data, H));

		Eigen::MatrixXd truth(3, 3);
		truth << -1, -1, -3,  8, 4, -6,  -12, -5,  8;

		BOOST_CHECK(((data - truth).array().abs() < Eigen::MatrixXd::Constant(3, 3, eps_).array()).all());

		// display.
		//std::cout << "data after normalization by homogeneous transformation = " << std::endl << data << std::endl;
		/*
		data after normalization by homogeneous transformation =
			-1  -1  -3
			8   4  -6
			-12  -5   8
		*/
	}

	// normalize data by Z-score.
	void testNormalizeDataByZScore()
	{
		Fixture fixture;

		Eigen::MatrixXd data(3, 3);
		data << 1, 2, 3,  4, 4, 4,  -1, 0, 3;  // row-major matrix.
		
		const double sigma = 1.0;
		DataNormalization::normalizeDataByZScore(data, sigma);

		Eigen::MatrixXd truth(3, 3);
		truth << -1, 0, 1,  0, 0, 0,  -1.66667, -0.666667, 2.33333;

		BOOST_CHECK(((data - truth).array().abs() < Eigen::MatrixXd::Constant(3, 3, eps_).array()).all());

		// display.
		//std::cout << "data after normalization by Z-score = " << std::endl << data << std::endl;
		/*
		data after normalization by Z-score =
			   -1         0         1
				0         0         0
		 -1.66667 -0.666667   2.33333
		*/
	}

	// normalize data by t-statistic.
	void testNormalizeDataByTStatistic()
	{
		Fixture fixture;

		Eigen::MatrixXd data(3, 3);
		data << 1, 2, 3,  4, 4, 4,  -1, 0, 3;  // row-major matrix.
		
		DataNormalization::normalizeDataByTStatistic(data);

		Eigen::MatrixXd truth(3, 3);
		truth << -1.73205, 0, 1.73205,  0, 0, 0,  -1.38675, -0.5547, 1.94145;

		BOOST_CHECK(((data - truth).array().abs() < Eigen::MatrixXd::Constant(3, 3, eps_).array()).all());

		// display.
		//std::cout << "data after normalization by t-statistic = " << std::endl << data << std::endl;
		/*
		data after normalization by t-statistic =
		-1.73205        0  1.73205
			   0        0        0
		-1.38675  -0.5547  1.94145
		*/
	}

private:
	const double eps_;
};

struct DataNormalizationTestSuite: public boost::unit_test_framework::test_suite
{
	DataNormalizationTestSuite()
	: boost::unit_test_framework::test_suite("SWL.Math.DataNormalization")
	{
		boost::shared_ptr<DataNormalizationTest> test(new DataNormalizationTest());

		add(BOOST_CLASS_TEST_CASE(&DataNormalizationTest::testNormalizeDataByCentering, test), 0);
		add(BOOST_CLASS_TEST_CASE(&DataNormalizationTest::testNormalizeDataByAverageDistance, test), 0);
		add(BOOST_CLASS_TEST_CASE(&DataNormalizationTest::testNormalizeDataByRange, test), 0);
		add(BOOST_CLASS_TEST_CASE(&DataNormalizationTest::testNormalizeDataByLinearTransformation, test), 0);
		add(BOOST_CLASS_TEST_CASE(&DataNormalizationTest::testNormalizeDataByHomogeneousTransformation, test), 0);
		add(BOOST_CLASS_TEST_CASE(&DataNormalizationTest::testNormalizeDataByZScore, test), 0);
		add(BOOST_CLASS_TEST_CASE(&DataNormalizationTest::testNormalizeDataByTStatistic, test), 0);

		boost::unit_test::framework::master_test_suite().add(this);
	}
} testsuite;

}  // unnamed namespace

//-----------------------------------------------------------------------------
//

#elif defined(__SWL_UNIT_TEST__USE_CPP_UNIT)

struct DataNormalizationTest: public CppUnit::TestFixture
{
private:
	CPPUNIT_TEST_SUITE(DataNormalizationTest);
	CPPUNIT_TEST(test);
	CPPUNIT_TEST_SUITE_END();

public:
	void setUp()  // set up
	{
	}

	void tearDown()  // tear down
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
//CPPUNIT_TEST_SUITE_REGISTRATION(swl::unit_test::DataNormalizationTest);
CPPUNIT_REGISTRY_ADD_TO_DEFAULT("SWL.Math");
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(swl::unit_test::DataNormalizationTest, "SWL.Math");
#endif
