#include "swl/Config.h"
#include "../../UnitTestConfig.h"
#include "swl/base/String.h"
#include "swl/util/RegionOfInterest.h"
#include <boost/smart_ptr.hpp>
#include <cstring>
#include <cmath>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace {
namespace local {

bool compareColors(const swl::RegionOfInterest::color_type &lhs, const swl::RegionOfInterest::color_type &rhs)
{
	const swl::RegionOfInterest::real_type eps = swl::RegionOfInterest::real_type(1.0e-15);
	return std::fabs(lhs.r - rhs.r) <= eps && std::fabs(lhs.g - rhs.g) <= eps && std::fabs(lhs.b - rhs.b) <= eps && std::fabs(lhs.a - rhs.a) <= eps;
}

}  // namespace local
}  // unnamed namespace

namespace swl {
namespace unit_test {

//-----------------------------------------------------------------------------
// Boost Test

#if defined(__SWL_UNIT_TEST__USE_BOOST_TEST)

namespace {

struct RegionOfInterestTest
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
	void testVisible()
	{
		Fixture fixture;

		boost::scoped_ptr<swl::RegionOfInterest> roi(new swl::LineROI(swl::LineROI::point_type(), swl::LineROI::point_type(), true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type()));
		BOOST_CHECK(roi);

		BOOST_CHECK(roi->isVisible());

		roi->setVisible(false);
		BOOST_CHECK(!roi->isVisible());

		roi->setVisible(true);
		BOOST_CHECK(roi->isVisible());
	}

	void testLineColor()
	{
		Fixture fixture;

		boost::scoped_ptr<swl::RegionOfInterest> roi(new swl::LineROI(swl::LineROI::point_type(), swl::LineROI::point_type(), true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type()));
		BOOST_CHECK(roi);

		BOOST_CHECK(local::compareColors(roi->getLineColor(), swl::RegionOfInterest::color_type()));

		// TODO [modify] >> erroneous case: 0.0f <= r, g, b, a <= 1.0f
		// but it's working normally
		roi->setLineColor(swl::RegionOfInterest::color_type(2.0f, 2.0f, 2.0f, 2.0f));
		BOOST_CHECK(local::compareColors(roi->getLineColor(), swl::RegionOfInterest::color_type(2.0f, 2.0f, 2.0f, 2.0f)));

		roi->setLineColor(swl::RegionOfInterest::color_type(0.1f, 0.2f, 0.8f, 0.9f));
		BOOST_CHECK(local::compareColors(roi->getLineColor(), swl::RegionOfInterest::color_type(0.1f, 0.2f, 0.8f, 0.9f)));
	}

	void testLineWidth()
	{
		Fixture fixture;

		boost::scoped_ptr<swl::RegionOfInterest> roi(new swl::LineROI(swl::LineROI::point_type(), swl::LineROI::point_type(), true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type()));
		BOOST_CHECK(roi);

		BOOST_CHECK_EQUAL(swl::RegionOfInterest::real_type(1), roi->getLineWidth());

		roi->setLineWidth(12.345f);
		BOOST_CHECK_EQUAL(swl::RegionOfInterest::real_type(12.345), roi->getLineWidth());
	}

	void testPointSize()
	{
		Fixture fixture;

		boost::scoped_ptr<swl::RegionOfInterest> roi(new swl::LineROI(swl::LineROI::point_type(), swl::LineROI::point_type(), true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type()));
		BOOST_CHECK(roi);

		BOOST_CHECK_EQUAL(swl::RegionOfInterest::real_type(1), roi->getPointSize());

		roi->setPointSize(5.302f);
		BOOST_CHECK_EQUAL(swl::RegionOfInterest::real_type(5.302f), roi->getPointSize());
	}

	void testPointColor()
	{
		Fixture fixture;

		boost::scoped_ptr<swl::RegionOfInterest> roi(new swl::LineROI(swl::LineROI::point_type(), swl::LineROI::point_type(), true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type()));
		BOOST_CHECK(roi);

		BOOST_CHECK(local::compareColors(roi->getPointColor(), swl::RegionOfInterest::color_type()));

		// TODO [modify] >> erroneous case: 0.0f <= r, g, b, a <= 1.0f
		// but it's working normally
		roi->setPointColor(swl::RegionOfInterest::color_type(6.0f, 7.0f, 8.0f, 1.0f));
		BOOST_CHECK(local::compareColors(roi->getPointColor(), swl::RegionOfInterest::color_type(6.0f, 7.0f, 8.0f, 1.0f)));

		roi->setPointColor(swl::RegionOfInterest::color_type(0.4f, 0.9f, 0.3f, 0.45f));
		BOOST_CHECK(local::compareColors(roi->getPointColor(), swl::RegionOfInterest::color_type(0.4f, 0.9f, 0.3f, 0.45f)));
	}

	void testName()
	{
		Fixture fixture;

		boost::scoped_ptr<swl::RegionOfInterest> roi(new swl::LineROI(swl::LineROI::point_type(), swl::LineROI::point_type(), true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type()));
		BOOST_CHECK(roi);

#if defined(UNICODE) || defined(_UNICODE)
		BOOST_CHECK(std::wcscmp(roi->getName().c_str(), L"") == 0);
		BOOST_CHECK(std::wcscmp(roi->getName().c_str(), std::wstring().c_str()) == 0);

		roi->setName(L"line ROI");
		BOOST_CHECK(std::wcscmp(roi->getName().c_str(), L"") != 0);
		BOOST_CHECK(std::wcscmp(roi->getName().c_str(), L"line ROI") == 0);
		BOOST_CHECK(std::wcscmp(roi->getName().c_str(), L"Line ROI") != 0);

		roi->setName(L"LINE_ROI");
		BOOST_CHECK(std::wcscmp(roi->getName().c_str(), L"LINE_ROI") == 0);
#else
		BOOST_CHECK(std::strcmp(roi->getName().c_str(), "") == 0);
		BOOST_CHECK(std::strcmp(roi->getName().c_str(), std::string().c_str()) == 0);

		roi->setName("line ROI");
		BOOST_CHECK(std::strcmp(roi->getName().c_str(), "") != 0);
		BOOST_CHECK(std::strcmp(roi->getName().c_str(), "line ROI") == 0);
		BOOST_CHECK(std::strcmp(roi->getName().c_str(), "Line ROI") != 0);

		roi->setName("LINE_ROI");
		BOOST_CHECK(std::strcmp(roi->getName().c_str(), "LINE_ROI") == 0);
#endif
	}
};

struct RegionOfInterestTestSuite: public boost::unit_test_framework::test_suite
{
	RegionOfInterestTestSuite()
	: boost::unit_test_framework::test_suite("SWL.Util.RegionOfInterest")
	{
		boost::shared_ptr<RegionOfInterestTest> test(new RegionOfInterestTest());

		add(BOOST_CLASS_TEST_CASE(&RegionOfInterestTest::testVisible, test), 0);
		add(BOOST_CLASS_TEST_CASE(&RegionOfInterestTest::testLineWidth, test), 0);
		add(BOOST_CLASS_TEST_CASE(&RegionOfInterestTest::testLineColor, test), 0);
		add(BOOST_CLASS_TEST_CASE(&RegionOfInterestTest::testPointSize, test), 0);
		add(BOOST_CLASS_TEST_CASE(&RegionOfInterestTest::testPointColor, test), 0);
		add(BOOST_CLASS_TEST_CASE(&RegionOfInterestTest::testName, test), 0);

		boost::unit_test::framework::master_test_suite().add(this);
	}
} testsuite;

}  // unnamed namespace

//-----------------------------------------------------------------------------
// Google Test

#elif defined(__SWL_UNIT_TEST__USE_GOOGLE_TEST)

class RegionOfInterestTest : public testing::Test
{
protected:
	/*virtual*/ void SetUp()
	{
	}

	/*virtual*/ void TearDown()
	{
	}
};

TEST_F(RegionOfInterestTest, testVisible)
{
	boost::scoped_ptr<swl::RegionOfInterest> roi(new swl::LineROI(swl::LineROI::point_type(), swl::LineROI::point_type(), true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type()));
	EXPECT_TRUE(roi);

	EXPECT_TRUE(roi->isVisible());

	roi->setVisible(false);
	EXPECT_FALSE(roi->isVisible());

	roi->setVisible(true);
	EXPECT_TRUE(roi->isVisible());
}

TEST_F(RegionOfInterestTest, testLineColor)
{
	boost::scoped_ptr<swl::RegionOfInterest> roi(new swl::LineROI(swl::LineROI::point_type(), swl::LineROI::point_type(), true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type()));
	EXPECT_TRUE(roi);

	EXPECT_TRUE(local::compareColors(roi->getLineColor(), swl::RegionOfInterest::color_type()));

	// TODO [modify] >> erroneous case: 0.0f <= r, g, b, a <= 1.0f
	// but it's working normally
	roi->setLineColor(swl::RegionOfInterest::color_type(2.0f, 2.0f, 2.0f, 2.0f));
	EXPECT_TRUE(local::compareColors(roi->getLineColor(), swl::RegionOfInterest::color_type(2.0f, 2.0f, 2.0f, 2.0f)));

	roi->setLineColor(swl::RegionOfInterest::color_type(0.1f, 0.2f, 0.8f, 0.9f));
	EXPECT_TRUE(local::compareColors(roi->getLineColor(), swl::RegionOfInterest::color_type(0.1f, 0.2f, 0.8f, 0.9f)));
}

TEST_F(RegionOfInterestTest, testLineWidth)
{
	boost::scoped_ptr<swl::RegionOfInterest> roi(new swl::LineROI(swl::LineROI::point_type(), swl::LineROI::point_type(), true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type()));
	EXPECT_TRUE(roi);

	EXPECT_EQ(swl::RegionOfInterest::real_type(1), roi->getLineWidth());

	roi->setLineWidth(12.345f);
	EXPECT_EQ(swl::RegionOfInterest::real_type(12.345), roi->getLineWidth());
}

TEST_F(RegionOfInterestTest, testPointSize)
{
	boost::scoped_ptr<swl::RegionOfInterest> roi(new swl::LineROI(swl::LineROI::point_type(), swl::LineROI::point_type(), true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type()));
	EXPECT_TRUE(roi);

	EXPECT_EQ(swl::RegionOfInterest::real_type(1), roi->getPointSize());

	roi->setPointSize(5.302f);
	EXPECT_EQ(swl::RegionOfInterest::real_type(5.302f), roi->getPointSize());
}

TEST_F(RegionOfInterestTest, testPointColor)
{
	boost::scoped_ptr<swl::RegionOfInterest> roi(new swl::LineROI(swl::LineROI::point_type(), swl::LineROI::point_type(), true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type()));
	EXPECT_TRUE(roi);

	EXPECT_TRUE(local::compareColors(roi->getPointColor(), swl::RegionOfInterest::color_type()));

	// TODO [modify] >> erroneous case: 0.0f <= r, g, b, a <= 1.0f
	// but it's working normally
	roi->setPointColor(swl::RegionOfInterest::color_type(6.0f, 7.0f, 8.0f, 1.0f));
	EXPECT_TRUE(local::compareColors(roi->getPointColor(), swl::RegionOfInterest::color_type(6.0f, 7.0f, 8.0f, 1.0f)));

	roi->setPointColor(swl::RegionOfInterest::color_type(0.4f, 0.9f, 0.3f, 0.45f));
	EXPECT_TRUE(local::compareColors(roi->getPointColor(), swl::RegionOfInterest::color_type(0.4f, 0.9f, 0.3f, 0.45f)));
}

TEST_F(RegionOfInterestTest, testName)
{
	boost::scoped_ptr<swl::RegionOfInterest> roi(new swl::LineROI(swl::LineROI::point_type(), swl::LineROI::point_type(), true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type()));
	EXPECT_TRUE(roi);

#if defined(UNICODE) || defined(_UNICODE)
	EXPECT_STREQ(roi->getName().c_str(), L"");
	EXPECT_STREQ(roi->getName().c_str(), std::wstring().c_str());

	roi->setName(L"line ROI");
	EXPECT_STRNE(roi->getName().c_str(), L"");
	EXPECT_STREQ(roi->getName().c_str(), L"line ROI");
	EXPECT_STRNE(roi->getName().c_str(), L"Line ROI");

	roi->setName(L"LINE_ROI");
	EXPECT_STREQ(roi->getName().c_str(), L"LINE_ROI");
#else
	EXPECT_STREQ(roi->getName().c_str(), "");
	EXPECT_STREQ(roi->getName().c_str(), std::string().c_str());

	roi->setName("line ROI");
	EXPECT_STRNE(roi->getName().c_str(), "");
	EXPECT_STREQ(roi->getName().c_str(), "line ROI");
	EXPECT_STRNE(roi->getName().c_str(), "Line ROI");

	roi->setName("LINE_ROI");
	EXPECT_STREQ(roi->getName().c_str(), "LINE_ROI");
#endif
}

//-----------------------------------------------------------------------------
// CppUnit

#elif defined(__SWL_UNIT_TEST__USE_CPP_UNIT)

struct RegionOfInterestTest: public CppUnit::TestFixture
{
private:
	CPPUNIT_TEST_SUITE(RegionOfInterestTest);
	CPPUNIT_TEST(testVisible);
	CPPUNIT_TEST(testLineWidth);
	CPPUNIT_TEST(testLineColor);
	CPPUNIT_TEST(testPointSize);
	CPPUNIT_TEST(testPointColor);
	CPPUNIT_TEST(testName);
	CPPUNIT_TEST_SUITE_END();

public:
	void setUp()  // set up
	{
	}

	void tearDown()  // tear down
	{
	}

	void testVisible()
	{
		boost::scoped_ptr<swl::RegionOfInterest> roi(new swl::LineROI(swl::LineROI::point_type(), swl::LineROI::point_type(), true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type()));
		CPPUNIT_ASSERT(roi);

		CPPUNIT_ASSERT(roi->isVisible());

		roi->setVisible(false);
		CPPUNIT_ASSERT(!roi->isVisible());

		roi->setVisible(true);
		CPPUNIT_ASSERT(roi->isVisible());
	}

	void testLineWidth()
	{
		boost::scoped_ptr<swl::RegionOfInterest> roi(new swl::LineROI(swl::LineROI::point_type(), swl::LineROI::point_type(), true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type()));
		CPPUNIT_ASSERT(roi);

		CPPUNIT_ASSERT_EQUAL(swl::RegionOfInterest::real_type(1), roi->getLineWidth());

		roi->setLineWidth(12.345f);
		CPPUNIT_ASSERT_EQUAL(swl::RegionOfInterest::real_type(12.345f), roi->getLineWidth());
	}

	void testLineColor()
	{
		boost::scoped_ptr<swl::RegionOfInterest> roi(new swl::LineROI(swl::LineROI::point_type(), swl::LineROI::point_type(), true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type()));
		CPPUNIT_ASSERT(roi);

		CPPUNIT_ASSERT(local::compareColors(roi->getLineColor(), swl::RegionOfInterest::color_type()));

		// TODO [modify] >> erroneous case: 0.0f <= r, g, b, a <= 1.0f
		// but it's working normally
		roi->setLineColor(swl::RegionOfInterest::color_type(2.0f, 2.0f, 2.0f, 2.0f));
		CPPUNIT_ASSERT(local::compareColors(roi->getLineColor(), swl::RegionOfInterest::color_type(2.0f, 2.0f, 2.0f, 2.0f)));

		roi->setLineColor(swl::RegionOfInterest::color_type(0.1f, 0.2f, 0.8f, 0.9f));
		CPPUNIT_ASSERT(local::compareColors(roi->getLineColor(), swl::RegionOfInterest::color_type(0.1f, 0.2f, 0.8f, 0.9f)));
	}

	void testPointSize()
	{
		boost::scoped_ptr<swl::RegionOfInterest> roi(new swl::LineROI(swl::LineROI::point_type(), swl::LineROI::point_type(), true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type()));
		CPPUNIT_ASSERT(roi);

		CPPUNIT_ASSERT_EQUAL(swl::RegionOfInterest::real_type(1), roi->getPointSize());

		roi->setPointSize(5.302f);
		CPPUNIT_ASSERT_EQUAL(swl::RegionOfInterest::real_type(5.302f), roi->getPointSize());
	}

	void testPointColor()
	{
		boost::scoped_ptr<swl::RegionOfInterest> roi(new swl::LineROI(swl::LineROI::point_type(), swl::LineROI::point_type(), true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type()));
		CPPUNIT_ASSERT(roi);

		CPPUNIT_ASSERT(local::compareColors(roi->getPointColor(), swl::RegionOfInterest::color_type()));

		// TODO [modify] >> erroneous case: 0.0f <= r, g, b, a <= 1.0f
		// but it's working normally
		roi->setPointColor(swl::RegionOfInterest::color_type(6.0f, 7.0f, 8.0f, 1.0f));
		CPPUNIT_ASSERT(local::compareColors(roi->getPointColor(), swl::RegionOfInterest::color_type(6.0f, 7.0f, 8.0f, 1.0f)));

		roi->setPointColor(swl::RegionOfInterest::color_type(0.4f, 0.9f, 0.3f, 0.45f));
		CPPUNIT_ASSERT(local::compareColors(roi->getPointColor(), swl::RegionOfInterest::color_type(0.4f, 0.9f, 0.3f, 0.45f)));
	}

	void testName()
	{
		boost::scoped_ptr<swl::RegionOfInterest> roi(new swl::LineROI(swl::LineROI::point_type(), swl::LineROI::point_type(), true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type()));
		CPPUNIT_ASSERT(roi);

#if defined(UNICODE) || defined(_UNICODE)
		CPPUNIT_ASSERT(std::wcscmp(roi->getName().c_str(), L"") == 0);
		CPPUNIT_ASSERT(std::wcscmp(roi->getName().c_str(), std::wstring().c_str()) == 0);

		roi->setName(L"line ROI");
		CPPUNIT_ASSERT(std::wcscmp(roi->getName().c_str(), L"") != 0);
		CPPUNIT_ASSERT(std::wcscmp(roi->getName().c_str(), L"line ROI") == 0);
		CPPUNIT_ASSERT(std::wcscmp(roi->getName().c_str(), L"Line ROI") != 0);

		roi->setName(L"LINE_ROI");
		CPPUNIT_ASSERT(std::wcscmp(roi->getName().c_str(), L"LINE_ROI") == 0);
#else
		CPPUNIT_ASSERT(std::strcmp(roi->getName().c_str(), "") == 0);
		CPPUNIT_ASSERT(std::strcmp(roi->getName().c_str(), std::string().c_str()) == 0);

		roi->setName("line ROI");
		CPPUNIT_ASSERT(std::strcmp(roi->getName().c_str(), "") != 0);
		CPPUNIT_ASSERT(std::strcmp(roi->getName().c_str(), "line ROI") == 0);
		CPPUNIT_ASSERT(std::strcmp(roi->getName().c_str(), "Line ROI") != 0);

		roi->setName("LINE_ROI");
		CPPUNIT_ASSERT(std::strcmp(roi->getName().c_str(), "LINE_ROI") == 0);
#endif
	}
};

#endif

}  // namespace unit_test
}  // namespace swl

#if defined(__SWL_UNIT_TEST__USE_CPP_UNIT)
//CPPUNIT_TEST_SUITE_REGISTRATION(swl::unit_test::RegionOfInterestTest);
CPPUNIT_REGISTRY_ADD_TO_DEFAULT("SWL.Util");
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(swl::unit_test::RegionOfInterestTest, "SWL.Util");
#endif
