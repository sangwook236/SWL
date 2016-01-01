#include "swl/Config.h"
#include "../../UnitTestConfig.h"
#include "swl/base/String.h"
#include "swl/util/RegionOfInterestMgr.h"
#include "swl/base/LogException.h"
#include <ctime>


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

struct RegionOfInterestMgrTest
{
private:
	struct Fixture
	{
		Fixture()  // set up
		{
			std::srand((unsigned int)std::time(NULL));
			swl::RegionOfInterestMgr::getInstance();
		}

		~Fixture()  // tear down
		{
			swl::RegionOfInterestMgr::clearInstance();
		}
	};

public:
	void testHandleROI()
	{
		Fixture fixture;

		swl::RegionOfInterestMgr &roiMgr = swl::RegionOfInterestMgr::getInstance();

		swl::LineROI roi1(swl::LineROI::point_type((swl::LineROI::real_type)std::rand(), (swl::LineROI::real_type)std::rand()), swl::LineROI::point_type((swl::LineROI::real_type)std::rand(), (swl::LineROI::real_type)std::rand()), true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type());
		swl::RectangleROI roi2(swl::RectangleROI::point_type((swl::RectangleROI::real_type)std::rand(), (swl::RectangleROI::real_type)std::rand()), swl::RectangleROI::point_type((swl::RectangleROI::real_type)std::rand(), (swl::RectangleROI::real_type)std::rand()), true, swl::RectangleROI::real_type(1), swl::RectangleROI::real_type(1), swl::RectangleROI::color_type(), swl::RectangleROI::color_type());
		swl::PolylineROI roi3(true, swl::PolylineROI::real_type(1), swl::PolylineROI::real_type(1), swl::PolylineROI::color_type(), swl::PolylineROI::color_type());
		for (int i = 0; i < ((std::rand() + 1) % 101); ++i)
			roi3.addPoint(swl::PolylineROI::point_type((swl::PolylineROI::real_type)std::rand(), (swl::PolylineROI::real_type)std::rand()));
		swl::PolygonROI roi4(true, swl::PolygonROI::real_type(1), swl::PolygonROI::real_type(1), swl::PolygonROI::color_type(), swl::PolygonROI::color_type());
		for (int i = 0; i < ((std::rand() + 1) % 101); ++i)
			roi4.addPoint(swl::PolygonROI::point_type((swl::PolygonROI::real_type)std::rand(), (swl::PolygonROI::real_type)std::rand()));

		BOOST_CHECK(roi3.containPoint());
		BOOST_CHECK(roi4.containPoint());
		const std::size_t countROI3 = roi3.countPoint();
		const std::size_t countROI4 = roi4.countPoint();

		BOOST_CHECK(!roiMgr.containROI());
		BOOST_CHECK_EQUAL((std::size_t)0, roiMgr.countROI());

		roiMgr.addROI(roi1);
		roiMgr.addROI(roi2);
		roiMgr.addROI(roi2);
		roiMgr.addROI(roi3);
		roiMgr.addROI(roi4);

		BOOST_CHECK(roiMgr.containROI());
		BOOST_CHECK_EQUAL((std::size_t)5, roiMgr.countROI());

		BOOST_CHECK(typeid(swl::LineROI) == typeid(roiMgr.getROI(0)));
		BOOST_CHECK(typeid(swl::RectangleROI) == typeid(roiMgr.getROI(1)));
		BOOST_CHECK(typeid(swl::RectangleROI) == typeid(roiMgr.getROI(2)));
		BOOST_CHECK(typeid(swl::PolylineROI) == typeid(roiMgr.getROI(3)));
		BOOST_CHECK(typeid(swl::PolygonROI) == typeid(roiMgr.getROI(4)));
		BOOST_CHECK(typeid(const swl::PolygonROI) == typeid(roiMgr.getROI(4)));

		BOOST_CHECK_THROW(roiMgr.getROI(5), swl::LogException);

		BOOST_CHECK_EQUAL(countROI3, dynamic_cast<swl::PolylineROI &>(roiMgr.getROI(3)).countPoint());
		BOOST_CHECK_EQUAL(countROI4, dynamic_cast<swl::PolygonROI &>(roiMgr.getROI(4)).countPoint());

		roiMgr.removeROI(3);
		roiMgr.removeROI(1);

		BOOST_CHECK(roiMgr.containROI());
		BOOST_CHECK_EQUAL((std::size_t)3, roiMgr.countROI());

		roiMgr.clearAllROIs();
		BOOST_CHECK(!roiMgr.containROI());
		BOOST_CHECK_EQUAL((std::size_t)0, roiMgr.countROI());
	}

	void testHandleValidRegion()
	{
		Fixture fixture;

		swl::RegionOfInterestMgr &roiMgr = swl::RegionOfInterestMgr::getInstance();

		BOOST_CHECK(roiMgr.isInValidRegion(swl::RegionOfInterestMgr::point_type((swl::RegionOfInterestMgr::real_type)std::rand(), (swl::RegionOfInterestMgr::real_type)std::rand())));

		roiMgr.setValidRegion(swl::RegionOfInterestMgr::region_type((swl::RegionOfInterestMgr::real_type)-25, (swl::RegionOfInterestMgr::real_type)10, (swl::RegionOfInterestMgr::real_type)75, (swl::RegionOfInterestMgr::real_type)50));
		BOOST_CHECK(roiMgr.isInValidRegion(swl::RegionOfInterestMgr::point_type((swl::RegionOfInterestMgr::real_type)0, (swl::RegionOfInterestMgr::real_type)20)));
		BOOST_CHECK(roiMgr.isInValidRegion(swl::RegionOfInterestMgr::point_type((swl::RegionOfInterestMgr::real_type)-5, (swl::RegionOfInterestMgr::real_type)37)));
		BOOST_CHECK(!roiMgr.isInValidRegion(swl::RegionOfInterestMgr::point_type((swl::RegionOfInterestMgr::real_type)-35, (swl::RegionOfInterestMgr::real_type)37)));
		BOOST_CHECK(!roiMgr.isInValidRegion(swl::RegionOfInterestMgr::point_type((swl::RegionOfInterestMgr::real_type)-5, (swl::RegionOfInterestMgr::real_type)65)));

		roiMgr.resetValidRegion();
		BOOST_CHECK(roiMgr.isInValidRegion(swl::RegionOfInterestMgr::point_type((swl::RegionOfInterestMgr::real_type)std::rand(), (swl::RegionOfInterestMgr::real_type)std::rand())));
		BOOST_CHECK(roiMgr.isInValidRegion(swl::RegionOfInterestMgr::point_type((swl::RegionOfInterestMgr::real_type)0, (swl::RegionOfInterestMgr::real_type)20)));
		BOOST_CHECK(roiMgr.isInValidRegion(swl::RegionOfInterestMgr::point_type((swl::RegionOfInterestMgr::real_type)-5, (swl::RegionOfInterestMgr::real_type)37)));
		BOOST_CHECK(roiMgr.isInValidRegion(swl::RegionOfInterestMgr::point_type((swl::RegionOfInterestMgr::real_type)-35, (swl::RegionOfInterestMgr::real_type)37)));
		BOOST_CHECK(roiMgr.isInValidRegion(swl::RegionOfInterestMgr::point_type((swl::RegionOfInterestMgr::real_type)-5, (swl::RegionOfInterestMgr::real_type)65)));
	}
};

struct RegionOfInterestMgrTestSuite: public boost::unit_test_framework::test_suite
{
	RegionOfInterestMgrTestSuite()
	: boost::unit_test_framework::test_suite("SWL.Util.RegionOfInterestMgr")
	{
		boost::shared_ptr<RegionOfInterestMgrTest> test(new RegionOfInterestMgrTest());

		add(BOOST_CLASS_TEST_CASE(&RegionOfInterestMgrTest::testHandleROI, test), 0);
		add(BOOST_CLASS_TEST_CASE(&RegionOfInterestMgrTest::testHandleValidRegion, test), 0);

		boost::unit_test::framework::master_test_suite().add(this);
	}
} testsuite;

}  // unnamed namespace

//-----------------------------------------------------------------------------
// Google Test

#elif defined(__SWL_UNIT_TEST__USE_GOOGLE_TEST)

class RegionOfInterestMgrTest : public testing::Test
{
protected:
	/*virtual*/ void SetUp()
	{
		std::srand((unsigned int)std::time(NULL));
		swl::RegionOfInterestMgr::getInstance();
	}

	/*virtual*/ void TearDown()
	{
		swl::RegionOfInterestMgr::clearInstance();
	}
};

TEST_F(RegionOfInterestMgrTest, testHandleROI)
{
	swl::RegionOfInterestMgr &roiMgr = swl::RegionOfInterestMgr::getInstance();

	swl::LineROI roi1(swl::LineROI::point_type((swl::LineROI::real_type)std::rand(), (swl::LineROI::real_type)std::rand()), swl::LineROI::point_type((swl::LineROI::real_type)std::rand(), (swl::LineROI::real_type)std::rand()), true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type());
	swl::RectangleROI roi2(swl::RectangleROI::point_type((swl::RectangleROI::real_type)std::rand(), (swl::RectangleROI::real_type)std::rand()), swl::RectangleROI::point_type((swl::RectangleROI::real_type)std::rand(), (swl::RectangleROI::real_type)std::rand()), true, swl::RectangleROI::real_type(1), swl::RectangleROI::real_type(1), swl::RectangleROI::color_type(), swl::RectangleROI::color_type());
	swl::PolylineROI roi3(true, swl::PolylineROI::real_type(1), swl::PolylineROI::real_type(1), swl::PolylineROI::color_type(), swl::PolylineROI::color_type());
	for (int i = 0; i < ((std::rand() + 1) % 101); ++i)
		roi3.addPoint(swl::PolylineROI::point_type((swl::PolylineROI::real_type)std::rand(), (swl::PolylineROI::real_type)std::rand()));
	swl::PolygonROI roi4(true, swl::PolygonROI::real_type(1), swl::PolygonROI::real_type(1), swl::PolygonROI::color_type(), swl::PolygonROI::color_type());
	for (int i = 0; i < ((std::rand() + 1) % 101); ++i)
		roi4.addPoint(swl::PolygonROI::point_type((swl::PolygonROI::real_type)std::rand(), (swl::PolygonROI::real_type)std::rand()));

	EXPECT_TRUE(roi3.containPoint());
	EXPECT_TRUE(roi4.containPoint());
	const std::size_t countROI3 = roi3.countPoint();
	const std::size_t countROI4 = roi4.countPoint();

	EXPECT_FALSE(roiMgr.containROI());
	EXPECT_EQ((std::size_t)0, roiMgr.countROI());

	roiMgr.addROI(roi1);
	roiMgr.addROI(roi2);
	roiMgr.addROI(roi2);
	roiMgr.addROI(roi3);
	roiMgr.addROI(roi4);

	EXPECT_TRUE(roiMgr.containROI());
	EXPECT_EQ((std::size_t)5, roiMgr.countROI());

	EXPECT_TRUE(typeid(swl::LineROI) == typeid(roiMgr.getROI(0)));
	EXPECT_TRUE(typeid(swl::RectangleROI) == typeid(roiMgr.getROI(1)));
	EXPECT_TRUE(typeid(swl::RectangleROI) == typeid(roiMgr.getROI(2)));
	EXPECT_TRUE(typeid(swl::PolylineROI) == typeid(roiMgr.getROI(3)));
	EXPECT_TRUE(typeid(swl::PolygonROI) == typeid(roiMgr.getROI(4)));
	EXPECT_TRUE(typeid(const swl::PolygonROI) == typeid(roiMgr.getROI(4)));

	EXPECT_THROW(roiMgr.getROI(5), swl::LogException);

	EXPECT_EQ(countROI3, dynamic_cast<swl::PolylineROI &>(roiMgr.getROI(3)).countPoint());
	EXPECT_EQ(countROI4, dynamic_cast<swl::PolygonROI &>(roiMgr.getROI(4)).countPoint());

	roiMgr.removeROI(3);
	roiMgr.removeROI(1);

	EXPECT_TRUE(roiMgr.containROI());
	EXPECT_EQ((std::size_t)3, roiMgr.countROI());

	roiMgr.clearAllROIs();
	EXPECT_TRUE(!roiMgr.containROI());
	EXPECT_EQ((std::size_t)0, roiMgr.countROI());
}

TEST_F(RegionOfInterestMgrTest, testHandleValidRegion)
{
	swl::RegionOfInterestMgr &roiMgr = swl::RegionOfInterestMgr::getInstance();

	EXPECT_TRUE(roiMgr.isInValidRegion(swl::RegionOfInterestMgr::point_type((swl::RegionOfInterestMgr::real_type)std::rand(), (swl::RegionOfInterestMgr::real_type)std::rand())));

	roiMgr.setValidRegion(swl::RegionOfInterestMgr::region_type((swl::RegionOfInterestMgr::real_type) - 25, (swl::RegionOfInterestMgr::real_type)10, (swl::RegionOfInterestMgr::real_type)75, (swl::RegionOfInterestMgr::real_type)50));
	EXPECT_TRUE(roiMgr.isInValidRegion(swl::RegionOfInterestMgr::point_type((swl::RegionOfInterestMgr::real_type)0, (swl::RegionOfInterestMgr::real_type)20)));
	EXPECT_TRUE(roiMgr.isInValidRegion(swl::RegionOfInterestMgr::point_type((swl::RegionOfInterestMgr::real_type) - 5, (swl::RegionOfInterestMgr::real_type)37)));
	EXPECT_TRUE(!roiMgr.isInValidRegion(swl::RegionOfInterestMgr::point_type((swl::RegionOfInterestMgr::real_type) - 35, (swl::RegionOfInterestMgr::real_type)37)));
	EXPECT_TRUE(!roiMgr.isInValidRegion(swl::RegionOfInterestMgr::point_type((swl::RegionOfInterestMgr::real_type) - 5, (swl::RegionOfInterestMgr::real_type)65)));

	roiMgr.resetValidRegion();
	EXPECT_TRUE(roiMgr.isInValidRegion(swl::RegionOfInterestMgr::point_type((swl::RegionOfInterestMgr::real_type)std::rand(), (swl::RegionOfInterestMgr::real_type)std::rand())));
	EXPECT_TRUE(roiMgr.isInValidRegion(swl::RegionOfInterestMgr::point_type((swl::RegionOfInterestMgr::real_type)0, (swl::RegionOfInterestMgr::real_type)20)));
	EXPECT_TRUE(roiMgr.isInValidRegion(swl::RegionOfInterestMgr::point_type((swl::RegionOfInterestMgr::real_type) - 5, (swl::RegionOfInterestMgr::real_type)37)));
	EXPECT_TRUE(roiMgr.isInValidRegion(swl::RegionOfInterestMgr::point_type((swl::RegionOfInterestMgr::real_type) - 35, (swl::RegionOfInterestMgr::real_type)37)));
	EXPECT_TRUE(roiMgr.isInValidRegion(swl::RegionOfInterestMgr::point_type((swl::RegionOfInterestMgr::real_type) - 5, (swl::RegionOfInterestMgr::real_type)65)));
}

//-----------------------------------------------------------------------------
// CppUnit

#elif defined(__SWL_UNIT_TEST__USE_CPP_UNIT)

struct RegionOfInterestMgrTest: public CppUnit::TestFixture
{
private:
	CPPUNIT_TEST_SUITE(RegionOfInterestMgrTest);
	CPPUNIT_TEST(testHandleROI);
	CPPUNIT_TEST(testHandleValidRegion);
	CPPUNIT_TEST_SUITE_END();

public:
	void setUp()  // set up
	{
		std::srand((unsigned int)std::time(NULL));
		swl::RegionOfInterestMgr::getInstance();
	}

	void tearDown()  // tear down
	{
		swl::RegionOfInterestMgr::clearInstance();
	}

	void testHandleROI()
	{
		swl::RegionOfInterestMgr &roiMgr = swl::RegionOfInterestMgr::getInstance();

		swl::LineROI roi1(swl::LineROI::point_type((swl::LineROI::real_type)std::rand(), (swl::LineROI::real_type)std::rand()), swl::LineROI::point_type((swl::LineROI::real_type)std::rand(), (swl::LineROI::real_type)std::rand()), true, swl::LineROI::real_type(1), swl::LineROI::real_type(1), swl::LineROI::color_type(), swl::LineROI::color_type());
		swl::RectangleROI roi2(swl::RectangleROI::point_type((swl::RectangleROI::real_type)std::rand(), (swl::RectangleROI::real_type)std::rand()), swl::RectangleROI::point_type((swl::RectangleROI::real_type)std::rand(), (swl::RectangleROI::real_type)std::rand()), true, swl::RectangleROI::real_type(1), swl::RectangleROI::real_type(1), swl::RectangleROI::color_type(), swl::RectangleROI::color_type());
		swl::PolylineROI roi3(true, swl::PolylineROI::real_type(1), swl::PolylineROI::real_type(1), swl::PolylineROI::color_type(), swl::PolylineROI::color_type());
		for (int i = 0; i < ((std::rand() + 1) % 101); ++i)
			roi3.addPoint(swl::PolylineROI::point_type((swl::PolylineROI::real_type)std::rand(), (swl::PolylineROI::real_type)std::rand()));
		swl::PolygonROI roi4(true, swl::PolygonROI::real_type(1), swl::PolygonROI::real_type(1), swl::PolygonROI::color_type(), swl::PolygonROI::color_type());
		for (int i = 0; i < ((std::rand() + 1) % 101); ++i)
			roi4.addPoint(swl::PolygonROI::point_type((swl::PolygonROI::real_type)std::rand(), (swl::PolygonROI::real_type)std::rand()));

		CPPUNIT_ASSERT(roi3.containPoint());
		CPPUNIT_ASSERT(roi4.containPoint());
		const std::size_t countROI3 = roi3.countPoint();
		const std::size_t countROI4 = roi4.countPoint();

		CPPUNIT_ASSERT(!roiMgr.containROI());
		CPPUNIT_ASSERT_EQUAL((std::size_t)0, roiMgr.countROI());

		roiMgr.addROI(roi1);
		roiMgr.addROI(roi2);
		roiMgr.addROI(roi2);
		roiMgr.addROI(roi3);
		roiMgr.addROI(roi4);

		CPPUNIT_ASSERT(roiMgr.containROI());
		CPPUNIT_ASSERT_EQUAL((std::size_t)5, roiMgr.countROI());

		CPPUNIT_ASSERT(typeid(swl::LineROI) == typeid(roiMgr.getROI(0)));
		CPPUNIT_ASSERT(typeid(swl::RectangleROI) == typeid(roiMgr.getROI(1)));
		CPPUNIT_ASSERT(typeid(swl::RectangleROI) == typeid(roiMgr.getROI(2)));
		CPPUNIT_ASSERT(typeid(swl::PolylineROI) == typeid(roiMgr.getROI(3)));
		CPPUNIT_ASSERT(typeid(swl::PolygonROI) == typeid(roiMgr.getROI(4)));
		CPPUNIT_ASSERT(typeid(const swl::PolygonROI) == typeid(roiMgr.getROI(4)));

		CPPUNIT_ASSERT_THROW(roiMgr.getROI(5), swl::LogException);

		CPPUNIT_ASSERT_EQUAL(countROI3, dynamic_cast<swl::PolylineROI &>(roiMgr.getROI(3)).countPoint());
		CPPUNIT_ASSERT_EQUAL(countROI4, dynamic_cast<swl::PolygonROI &>(roiMgr.getROI(4)).countPoint());

		roiMgr.removeROI(3);
		roiMgr.removeROI(1);

		CPPUNIT_ASSERT(roiMgr.containROI());
		CPPUNIT_ASSERT_EQUAL((std::size_t)3, roiMgr.countROI());

		roiMgr.clearAllROIs();
		CPPUNIT_ASSERT(!roiMgr.containROI());
		CPPUNIT_ASSERT_EQUAL((std::size_t)0, roiMgr.countROI());
	}

	void testHandleValidRegion()
	{
		swl::RegionOfInterestMgr &roiMgr = swl::RegionOfInterestMgr::getInstance();

		CPPUNIT_ASSERT(roiMgr.isInValidRegion(swl::RegionOfInterestMgr::point_type((swl::RegionOfInterestMgr::real_type)std::rand(), (swl::RegionOfInterestMgr::real_type)std::rand())));

		roiMgr.setValidRegion(swl::RegionOfInterestMgr::region_type((swl::RegionOfInterestMgr::real_type)-25, (swl::RegionOfInterestMgr::real_type)10, (swl::RegionOfInterestMgr::real_type)75, (swl::RegionOfInterestMgr::real_type)50));
		CPPUNIT_ASSERT(roiMgr.isInValidRegion(swl::RegionOfInterestMgr::point_type((swl::RegionOfInterestMgr::real_type)0, (swl::RegionOfInterestMgr::real_type)20)));
		CPPUNIT_ASSERT(roiMgr.isInValidRegion(swl::RegionOfInterestMgr::point_type((swl::RegionOfInterestMgr::real_type)-5, (swl::RegionOfInterestMgr::real_type)37)));
		CPPUNIT_ASSERT(!roiMgr.isInValidRegion(swl::RegionOfInterestMgr::point_type((swl::RegionOfInterestMgr::real_type)-35, (swl::RegionOfInterestMgr::real_type)37)));
		CPPUNIT_ASSERT(!roiMgr.isInValidRegion(swl::RegionOfInterestMgr::point_type((swl::RegionOfInterestMgr::real_type)-5, (swl::RegionOfInterestMgr::real_type)65)));

		roiMgr.resetValidRegion();
		CPPUNIT_ASSERT(roiMgr.isInValidRegion(swl::RegionOfInterestMgr::point_type((swl::RegionOfInterestMgr::real_type)std::rand(), (swl::RegionOfInterestMgr::real_type)std::rand())));
		CPPUNIT_ASSERT(roiMgr.isInValidRegion(swl::RegionOfInterestMgr::point_type((swl::RegionOfInterestMgr::real_type)0, (swl::RegionOfInterestMgr::real_type)20)));
		CPPUNIT_ASSERT(roiMgr.isInValidRegion(swl::RegionOfInterestMgr::point_type((swl::RegionOfInterestMgr::real_type)-5, (swl::RegionOfInterestMgr::real_type)37)));
		CPPUNIT_ASSERT(roiMgr.isInValidRegion(swl::RegionOfInterestMgr::point_type((swl::RegionOfInterestMgr::real_type)-35, (swl::RegionOfInterestMgr::real_type)37)));
		CPPUNIT_ASSERT(roiMgr.isInValidRegion(swl::RegionOfInterestMgr::point_type((swl::RegionOfInterestMgr::real_type)-5, (swl::RegionOfInterestMgr::real_type)65)));
	}
};

#endif

}  // namespace unit_test
}  // namespace swl

#if defined(__SWL_UNIT_TEST__USE_CPP_UNIT)
//CPPUNIT_TEST_SUITE_REGISTRATION(swl::unit_test::RegionOfInterestMgrTest);
CPPUNIT_REGISTRY_ADD_TO_DEFAULT("SWL.Util");
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(swl::unit_test::RegionOfInterestMgrTest, "SWL.Util");
#endif
