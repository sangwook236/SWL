#include "swl/Config.h"
#include "../../UnitTestConfig.h"
#include "swl/base/String.h"
#include "swl/math/MathUtil.h"


#if defined(_DEBUG)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {
namespace unit_test {

//-----------------------------------------------------------------------------
//

#if defined(__SWL_UNIT_TEST__USE_BOOST_UNIT)

namespace {

struct BaseFieldConversionTest
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
	void testDec2Bin()
	{
		Fixture fixture;

		const swl::string_t num = MathUtil::dec2bin(432ul);
		BOOST_CHECK_EQUAL(num, SWL_STR("110110000"));
		BOOST_CHECK_NE(num, SWL_STR("110101111"));
		BOOST_CHECK_NE(num, SWL_STR("110110001"));

		const swl::string_t num1 = MathUtil::dec2bin(-3796ul);
		const swl::string_t num2 = MathUtil::dec2bin(0xFFFFFFFFul - 3796ul + 1ul);  // 2's complement of a binary number
		BOOST_CHECK_EQUAL(num1, num2);
	}

	void testDec2Oct()
	{
		Fixture fixture;

		const swl::string_t num = MathUtil::dec2oct(432ul);
		BOOST_CHECK_EQUAL(num, SWL_STR("660"));
		BOOST_CHECK_NE(num, SWL_STR("657"));
		BOOST_CHECK_NE(num, SWL_STR("661"));

		const swl::string_t num1 = MathUtil::dec2oct(-3796ul);
		const swl::string_t num2 = MathUtil::dec2oct(0xFFFFFFFFul - 3796ul + 1ul);  // 2's complement of a binary number
		BOOST_CHECK_EQUAL(num1, num2);
	}

	void testDec2Hex()
	{
		Fixture fixture;

		const swl::string_t num = MathUtil::dec2hex(432ul);
		BOOST_CHECK_EQUAL(num, SWL_STR("1B0"));
		BOOST_CHECK_NE(num, SWL_STR("1AF"));
		BOOST_CHECK_NE(num, SWL_STR("1B1"));

		const swl::string_t num1 = MathUtil::dec2hex(-3796ul);
		const swl::string_t num2 = MathUtil::dec2hex(0xFFFFFFFFul - 3796ul + 1ul);  // 2's complement of a binary number
		BOOST_CHECK_EQUAL(num1, num2);
	}

	void testBin2Dec()
	{
		Fixture fixture;

		const unsigned long dec = MathUtil::bin2dec(SWL_STR("110110000"));
		BOOST_CHECK_EQUAL(dec, 432ul);
		BOOST_CHECK_NE(dec, 431ul);
		BOOST_CHECK_NE(dec, 433ul);
	}

	void testOct2Dec()
	{
		Fixture fixture;

		const unsigned long dec = MathUtil::oct2dec(SWL_STR("660"));
		BOOST_CHECK_EQUAL(dec, 432ul);
		BOOST_CHECK_NE(dec, 431ul);
		BOOST_CHECK_NE(dec, 433ul);
	}

	void testHex2Dec()
	{
		Fixture fixture;

		const unsigned long dec = MathUtil::hex2dec(SWL_STR("1B0"));
		BOOST_CHECK_EQUAL(dec, 432ul);
		BOOST_CHECK_NE(dec, 431ul);
		BOOST_CHECK_NE(dec, 433ul);
	}
};

struct BaseFieldConversionTestSuite: public boost::unit_test_framework::test_suite
{
	BaseFieldConversionTestSuite()
	: boost::unit_test_framework::test_suite("SWL.Math.BaseFieldConversion")
	{
		boost::shared_ptr<BaseFieldConversionTest> test(new BaseFieldConversionTest());

		add(BOOST_CLASS_TEST_CASE(&BaseFieldConversionTest::testDec2Bin, test), 0);
		add(BOOST_CLASS_TEST_CASE(&BaseFieldConversionTest::testDec2Oct, test), 0);
		add(BOOST_CLASS_TEST_CASE(&BaseFieldConversionTest::testDec2Hex, test), 0);
		add(BOOST_CLASS_TEST_CASE(&BaseFieldConversionTest::testBin2Dec, test), 0);
		add(BOOST_CLASS_TEST_CASE(&BaseFieldConversionTest::testOct2Dec, test), 0);
		add(BOOST_CLASS_TEST_CASE(&BaseFieldConversionTest::testHex2Dec, test), 0);

		boost::unit_test::framework::master_test_suite().add(this);
	}
} testsuite;

}  // unnamed namespace

//-----------------------------------------------------------------------------
//

#elif defined(__SWL_UNIT_TEST__USE_CPP_UNIT)

struct BaseFieldConversionTest: public CppUnit::TestFixture
{
private:
	CPPUNIT_TEST_SUITE(BaseFieldConversionTest);
	CPPUNIT_TEST(testDec2Bin);
	CPPUNIT_TEST(testDec2Oct);
	CPPUNIT_TEST(testDec2Hex);
	CPPUNIT_TEST(testBin2Dec);
	CPPUNIT_TEST(testOct2Dec);
	CPPUNIT_TEST(testHex2Dec);
	CPPUNIT_TEST_SUITE_END();

public:
	void setUp()  // set up
	{
	}

	void tearDown()  // tear down
	{
	}

	void testDec2Bin()
	{
		const swl::string_t num = MathUtil::dec2bin(432ul);
		CPPUNIT_ASSERT(num == SWL_STR("110110000"));
		CPPUNIT_ASSERT(num != SWL_STR("110101111"));
		CPPUNIT_ASSERT(num != SWL_STR("110110001"));

		const swl::string_t num1 = MathUtil::dec2bin(-3796ul);
		const swl::string_t num2 = MathUtil::dec2bin(0xFFFFFFFFul - 3796ul + 1ul);  // 2's complement of a binary number
		CPPUNIT_ASSERT(num1 == num2);
	}

	void testDec2Oct()
	{
		const swl::string_t num = MathUtil::dec2oct(432ul);
		CPPUNIT_ASSERT(num == SWL_STR("660"));
		CPPUNIT_ASSERT(num != SWL_STR("657"));
		CPPUNIT_ASSERT(num != SWL_STR("661"));

		const swl::string_t num1 = MathUtil::dec2oct(-3796ul);
		const swl::string_t num2 = MathUtil::dec2oct(0xFFFFFFFFul - 3796ul + 1ul);  // 2's complement of a binary number
		CPPUNIT_ASSERT(num1 == num2);
	}

	void testDec2Hex()
	{
		const swl::string_t num = MathUtil::dec2hex(432ul);
		CPPUNIT_ASSERT(num == SWL_STR("1B0"));
		CPPUNIT_ASSERT(num != SWL_STR("1AF"));
		CPPUNIT_ASSERT(num != SWL_STR("1B1"));

		const swl::string_t num1 = MathUtil::dec2hex(-3796ul);
		const swl::string_t num2 = MathUtil::dec2hex(0xFFFFFFFFul - 3796ul + 1ul);  // 2's complement of a binary number
		CPPUNIT_ASSERT(num1 == num2);
	}

	void testBin2Dec()
	{
		const unsigned long dec = MathUtil::bin2dec(SWL_STR("110110000"));
		CPPUNIT_ASSERT_EQUAL(dec, 432ul);
		CPPUNIT_ASSERT(dec != 431ul);
		CPPUNIT_ASSERT(dec != 433ul);
	}

	void testOct2Dec()
	{
		const unsigned long dec = MathUtil::oct2dec(SWL_STR("660"));
		CPPUNIT_ASSERT_EQUAL(dec, 432ul);
		CPPUNIT_ASSERT(dec != 431ul);
		CPPUNIT_ASSERT(dec != 433ul);
	}

	void testHex2Dec()
	{
		const unsigned long dec = MathUtil::hex2dec(SWL_STR("1B0"));
		CPPUNIT_ASSERT_EQUAL(dec, 432ul);
		CPPUNIT_ASSERT(dec != 431ul);
		CPPUNIT_ASSERT(dec != 433ul);
	}
};

#endif

}  // namespace unit_test
}  // namespace swl

#if defined(__SWL_UNIT_TEST__USE_CPP_UNIT)
CPPUNIT_TEST_SUITE_REGISTRATION(swl::unit_test::BaseFieldConversionTest);
//CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(swl::unit_test::BaseFieldConversionTest, "SWL.Math.BaseFieldConversion");  // not working
#endif
