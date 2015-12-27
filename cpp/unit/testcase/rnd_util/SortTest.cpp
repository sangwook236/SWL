#include "swl/Config.h"
#include "../../UnitTestConfig.h"
#include "swl/rnd_util/Sort.h"


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

struct SortTest
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
	void testMergeSort()
	{
		Fixture fixture;

		std::vector<int> data1 = { 1, -1, 3, 10, 23, 37, 17, -5, -19, 2, 3, 81 };
		std::vector<int> data2(data1);

		std::sort(data1.begin(), data1.end());
		swl::Sort::mergeSort(data2.begin(), data2.end());

		std::vector<int>::iterator it = data2.begin();
		for (auto val : data1)
			BOOST_CHECK_EQUAL(val, *it++);
	}

	void testQuickSort()
	{
		Fixture fixture;

		std::vector<int> data1 = { 1, -1, 3, 10, 23, 37, 17, -5, -19, 2, 3, 81 };
		std::vector<int> data2(data1);

		std::sort(data1.begin(), data1.end());
		swl::Sort::quickSort(data2.begin(), data2.end());

		std::vector<int>::iterator it = data2.begin();
		for (auto val : data1)
			BOOST_CHECK_EQUAL(val, *it++);
	}
};

struct SortTestSuite : public boost::unit_test_framework::test_suite
{
	SortTestSuite()
	: boost::unit_test_framework::test_suite("SWL.RndUtil.Sort")
	{
		boost::shared_ptr<SortTest> test(new SortTest());

		add(BOOST_CLASS_TEST_CASE(&SortTest::testMergeSort, test), 0);
		add(BOOST_CLASS_TEST_CASE(&SortTest::testQuickSort, test), 0);

		boost::unit_test::framework::master_test_suite().add(this);
	}
} testsuite;

}  // unnamed namespace

//-----------------------------------------------------------------------------
//

#elif defined(__SWL_UNIT_TEST__USE_CPP_UNIT)

struct SortTest : public CppUnit::TestFixture
{
private:
	CPPUNIT_TEST_SUITE(SortTest);
	CPPUNIT_TEST(testMergeSort);
	CPPUNIT_TEST(testQuickSort);
	CPPUNIT_TEST_SUITE_END();

public:
	void setUp()  // set up
	{
	}

	void tearDown()  // tear down
	{
	}

	void testMergeSort()
	{
		std::vector<int> data1 = { 1, -1, 3, 10, 23, 37, 17, -5, -19, 2, 3, 81 };
		std::vector<int> data2(data1);

		std::sort(data1.begin(), data1.end());
		swl::Sort::mergeSort(data2.begin(), data2.end());

		std::vector<int>::iterator it = data2.begin();
		for (auto val : data1)
			CPPUNIT_ASSERT_EQUAL(val, *it++);
	}

	void testQuickSort()
	{
		std::vector<int> data1 = { 1, -1, 3, 10, 23, 37, 17, -5, -19, 2, 3, 81 };
		std::vector<int> data2(data1);

		std::sort(data1.begin(), data1.end());
		swl::Sort::quickSort(data2.begin(), data2.end());

		std::vector<int>::iterator it = data2.begin();
		for (auto val : data1)
			CPPUNIT_ASSERT_EQUAL(val, *it++);
	}
};

#endif

}  // namespace unit_test
}  // namespace swl

#if defined(__SWL_UNIT_TEST__USE_CPP_UNIT)
//CPPUNIT_TEST_SUITE_REGISTRATION(swl::unit_test::SortTest);
CPPUNIT_REGISTRY_ADD_TO_DEFAULT("SWL.RndUtil");
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(swl::unit_test::SortTest, "SWL.RndUtil");
#endif
