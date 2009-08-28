#include "swl/Config.h"
#include "../../UnitTestConfig.h"

//#define __SWL__DISABLE_LOGGER_ 1
#include "swl/base/Logger.h"


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

struct LoggerTest
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
	void testBasic()
	{
		Fixture fixture;

		LOG4CXX_WARN(logger_, L"Low fuel level.");
		LOG4CXX_ERROR(tracer_, L"Located nearest gas station.");

		logger_->setLevel(log4cxx::Level::getInfo());
		BOOST_CHECK_EQUAL(log4cxx::Level::getInfo(), logger_->getLevel());

		LOG4CXX_DEBUG(logger_, L"Starting search for nearest gas station.");
		LOG4CXX_DEBUG(tracer_, L"Exiting gas station search.");
	}

public:
	LoggerTest()
	: logger_(log4cxx::Logger::getLogger(L"swlLogger.logger")), tracer_(log4cxx::Logger::getLogger(L"swlLogger.tracer"))
	{
		const int config = 1;
		switch (config)
		{
		case 1:
			log4cxx::PropertyConfigurator::configure(L"test_data\\swl_logger_conf.properties");
			break;
		case 2:
			log4cxx::xml::DOMConfigurator::configure(L"test_data\\swl_logger_conf.xml");
			break;
		case 0:
		default:
			log4cxx::BasicConfigurator::configure();
			break;
		}
	}

private:
	log4cxx::LoggerPtr logger_;
	log4cxx::LoggerPtr tracer_;
};

struct LoggerTestSuite: public boost::unit_test_framework::test_suite
{
	LoggerTestSuite()
	: boost::unit_test_framework::test_suite("SWL.Base.Logger")
	{
		boost::shared_ptr<LoggerTest> test(new LoggerTest());

		add(BOOST_CLASS_TEST_CASE(&LoggerTest::testBasic, test), 0);

		boost::unit_test::framework::master_test_suite().add(this);
	}
} testsuite;

}  // unnamed namespace

//-----------------------------------------------------------------------------
//

#elif defined(__SWL_UNIT_TEST__USE_CPP_UNIT)

struct LoggerTest: public CppUnit::TestFixture
{
private:
	CPPUNIT_TEST_SUITE(LoggerTest);
	CPPUNIT_TEST(testBasic);
	CPPUNIT_TEST_SUITE_END();

public:
	void setUp()  // set up
	{
	}

	void tearDown()  // tear down
	{
	}

	void testBasic()
	{
		LOG4CXX_WARN(logger_, L"Low fuel level.");
		LOG4CXX_ERROR(tracer_, L"Located nearest gas station.");

		logger_->setLevel(log4cxx::Level::getInfo());
		CPPUNIT_ASSERT_EQUAL(log4cxx::Level::getInfo(), logger_->getLevel());

		LOG4CXX_DEBUG(logger_, L"Starting search for nearest gas station.");
		LOG4CXX_DEBUG(tracer_, L"Exiting gas station search.");
	}

public:
	LoggerTest()
	: logger_(log4cxx::Logger::getLogger(L"swlLogger.logger")), tracer_(log4cxx::Logger::getLogger(L"swlLogger.tracer"))
	{
		const int config = 1;
		switch (config)
		{
		case 1:
			log4cxx::PropertyConfigurator::configure(L"test_data\\swl_logger_conf.properties");
			break;
		case 2:
			log4cxx::xml::DOMConfigurator::configure(L"test_data\\swl_logger_conf.xml");
			break;
		case 0:
		default:
			log4cxx::BasicConfigurator::configure();
			break;
		}
	}

private:
	log4cxx::LoggerPtr logger_;
	log4cxx::LoggerPtr tracer_;
};

#endif

}  // namespace unit_test
}  // namespace swl

#if defined(__SWL_UNIT_TEST__USE_CPP_UNIT)
CPPUNIT_TEST_SUITE_REGISTRATION(swl::unit_test::LoggerTest);
//CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(swl::unit_test::LoggerTest, "SWL.Base.Logger");  // not working
#endif
