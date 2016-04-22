#include "swl/Config.h"
#include "../../UnitTestConfig.h"
#include "swl/math/MathUtil.h"
#include "swl/base/String.h"
#include <stdexcept>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {
namespace unit_test {

//-----------------------------------------------------------------------------
// Boost Test

#if defined(__SWL_UNIT_TEST__USE_BOOST_TEST)

namespace {

struct MathUtilTest
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
	void testToPrecedingOdd()
	{
		Fixture fixture;

		const int inputs[] = { -10, -9, -8, -7, -6, -5, -4, -3, -2, -1,  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
		const int truths[] = { -11, -9, -9, -7, -7, -5, -5, -3, -3, -1, -1, 1, 1, 3, 3, 5, 5, 7, 7, 9,  9 };

		int i = 0;
		for (auto num : inputs)
			BOOST_CHECK_EQUAL(truths[i++], swl::MathUtil::toPrecedingOdd(num));
	}

	void testToFollowingOdd()
	{
		Fixture fixture;

		const int inputs[] = { -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
		const int truths[] = { -9, -9, -7, -7, -5, -5, -3, -3, -1, -1, 1, 1, 3, 3, 5, 5, 7, 7, 9, 9, 11 };

		int i = 0;
		for (auto num : inputs)
			BOOST_CHECK_EQUAL(truths[i++], swl::MathUtil::toFollowingOdd(num));
	}

	void testToPrecedingEven()
	{
		Fixture fixture;

		const int inputs[] = { -10,  -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
		const int truths[] = { -10, -10, -8, -8, -6, -6, -4, -4, -2, -2, 0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10 };

		int i = 0;
		for (auto num : inputs)
			BOOST_CHECK_EQUAL(truths[i++], swl::MathUtil::toPrecedingEven(num));
	}

	void testToFollowingEven()
	{
		Fixture fixture;

		const int inputs[] = { -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8,  9, 10 };
		const int truths[] = { -10, -8, -8, -6, -6, -4, -4, -2, -2,  0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10 };

		int i = 0;
		for (auto num : inputs)
			BOOST_CHECK_EQUAL(truths[i++], swl::MathUtil::toFollowingEven(num));
	}
};

struct MathUtilTestSuite : public boost::unit_test_framework::test_suite
{
	MathUtilTestSuite()
	: boost::unit_test_framework::test_suite("SWL.Math.MathUtil")
	{
		boost::shared_ptr<MathUtilTest> test(new MathUtilTest());

		add(BOOST_CLASS_TEST_CASE(&MathUtilTest::testToPrecedingOdd, test), 0);

		boost::unit_test::framework::master_test_suite().add(this);
	}
} testsuite;

}  // unnamed namespace

//-----------------------------------------------------------------------------
// Google Test

#elif defined(__SWL_UNIT_TEST__USE_GOOGLE_TEST)

class MathUtilTest : public testing::Test
{
protected:
	/*virtual*/ void SetUp()
	{
	}

	/*virtual*/ void TearDown()
	{
	}
};

TEST_F(MathUtilTest, testToPrecedingOdd)
{
	const int inputs[] = { -10, -9, -8, -7, -6, -5, -4, -3, -2, -1,  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
	const int truths[] = { -11, -9, -9, -7, -7, -5, -5, -3, -3, -1, -1, 1, 1, 3, 3, 5, 5, 7, 7, 9,  9 };

	int i = 0;
	for (auto num : inputs)
		EXPECT_EQ(truths[i++], swl::MathUtil::toPrecedingOdd(num));
}

TEST_F(MathUtilTest, testToFollowingOdd)
{
	const int inputs[] = { -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
	const int truths[] = {  -9, -9, -7, -7, -5, -5, -3, -3, -1, -1, 1, 1, 3, 3, 5, 5, 7, 7, 9, 9, 11 };

	int i = 0;
	for (auto num : inputs)
		EXPECT_EQ(truths[i++], swl::MathUtil::toFollowingOdd(num));
}

TEST_F(MathUtilTest, testToPrecedingEven)
{
	const int inputs[] = { -10,  -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
	const int truths[] = { -10, -10, -8, -8, -6, -6, -4, -4, -2, -2, 0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10 };

	int i = 0;
	for (auto num : inputs)
		EXPECT_EQ(truths[i++], swl::MathUtil::toPrecedingEven(num));
}

TEST_F(MathUtilTest, testToFollowingEven)
{
	const int inputs[] = { -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8,  9, 10 };
	const int truths[] = { -10, -8, -8, -6, -6, -4, -4, -2, -2,  0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10 };

	int i = 0;
	for (auto num : inputs)
		EXPECT_EQ(truths[i++], swl::MathUtil::toFollowingEven(num));
}

//-----------------------------------------------------------------------------
// CppUnit

#elif defined(__SWL_UNIT_TEST__USE_CPP_UNIT)

struct MathUtilTest : public CppUnit::TestFixture
{
private:
	CPPUNIT_TEST_SUITE(MathUtilTest);
	CPPUNIT_TEST(testToPrecedingOdd);
	CPPUNIT_TEST_SUITE_END();

public:
	void setUp()  // set up
	{
	}

	void tearDown()  // tear down
	{
	}

	void testToPrecedingOdd()
	{
		const int inputs[] = { -10, -9, -8, -7, -6, -5, -4, -3, -2, -1,  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
		const int truths[] = { -11, -9, -9, -7, -7, -5, -5, -3, -3, -1, -1, 1, 1, 3, 3, 5, 5, 7, 7, 9,  9 };

		int i = 0;
		for (auto num : inputs)
			CPPUNIT_ASSERT_EQUAL(truths[i++], swl::MathUtil::toPrecedingOdd(num));
	}

	void testToFollowingOdd()
	{
		const int inputs[] = { -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
		const int truths[] = { -9, -9, -7, -7, -5, -5, -3, -3, -1, -1, 1, 1, 3, 3, 5, 5, 7, 7, 9, 9, 11 };

		int i = 0;
		for (auto num : inputs)
			CPPUNIT_ASSERT_EQUAL(truths[i++], swl::MathUtil::toFollowingOdd(num));
	}

	void testToPrecedingEven()
	{
		const int inputs[] = { -10,  -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
		const int truths[] = { -10, -10, -8, -8, -6, -6, -4, -4, -2, -2, 0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10 };

		int i = 0;
		for (auto num : inputs)
			CPPUNIT_ASSERT_EQUAL(truths[i++], swl::MathUtil::toPrecedingEven(num));
	}

	void testToFollowingEven()
	{
		const int inputs[] = { -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8,  9, 10 };
		const int truths[] = { -10, -8, -8, -6, -6, -4, -4, -2, -2,  0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10 };

		int i = 0;
		for (auto num : inputs)
			CPPUNIT_ASSERT_EQUAL(truths[i++], swl::MathUtil::toFollowingEven(num));
	}
};

#endif

}  // namespace unit_test
}  // namespace swl

#if defined(__SWL_UNIT_TEST__USE_CPP_UNIT)
//CPPUNIT_TEST_SUITE_REGISTRATION(swl::unit_test::MathUtilTest);
CPPUNIT_REGISTRY_ADD_TO_DEFAULT("SWL.Math");
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(swl::unit_test::MathUtilTest, "SWL.Math");
#endif
