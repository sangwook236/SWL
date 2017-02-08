#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
#include <vld/vld.h>
#endif
#include "swl/Config.h"
#include "../UnitTestConfig.h"

//-----------------------------------------------------------------------------
// Boost Test.

#if defined(__SWL_UNIT_TEST__USE_BOOST_TEST)

#include <boost/test/unit_test_parameters.hpp>
#include <boost/test/unit_test.hpp>
#include <fstream>

//-----------------------------------------------------------------------------
// Google Test.

#elif defined(__SWL_UNIT_TEST__USE_GOOGLE_TEST)

//#include <gmock/gmock.h>
#include <gtest/gtest.h>

//-----------------------------------------------------------------------------
// CppUnit.

#elif defined(__SWL_UNIT_TEST__USE_CPP_UNIT)

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/ui/text/TestRunner.h>

#endif

#include <iostream>
#include <cstdlib>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


//-----------------------------------------------------------------------------
// Boost Test.

#if defined(__SWL_UNIT_TEST__USE_BOOST_TEST)

namespace {
namespace local {

// Runtime configration: command line arguments
//	--show_progress=yes --log_level=message --run_test=*

struct BoostUnitGlobalFixture
{
    BoostUnitGlobalFixture()
	: logStream_("swl_unittest.log")
	{
		boost::unit_test::unit_test_log.set_stream(logStream_);
		boost::unit_test::unit_test_log.set_format(boost::unit_test::OF_XML);
		//boost::unit_test::unit_test_log.set_formatter();
		if (boost::unit_test::runtime_config::log_level() < boost::unit_test::log_warnings)
			boost::unit_test::unit_test_log.set_threshold_level(boost::unit_test::log_warnings);
		//boost::unit_test::progress_monitor.set_stream(logStream_)
	}
    ~BoostUnitGlobalFixture()
	{
		boost::unit_test::unit_test_log.set_stream(std::cout);
	}

private:
    std::ofstream logStream_;
};

}  // namespace local
}  // unnamed namespace

//BOOST_GLOBAL_FIXTURE(local::BoostUnitGlobalFixture);

#if !defined(BOOST_TEST_MODULE)
boost::unit_test_framework::test_suite * init_unit_test_suite(int, char *[])
{
	//boost::unit_test::framework::master_test_suite().add(...);

	return 0;
}
#endif

//-----------------------------------------------------------------------------
// Google Test.

#elif defined(__SWL_UNIT_TEST__USE_GOOGLE_TEST)

int main(int argc, char *argv[])
{
	int retval = EXIT_SUCCESS;
	try
	{
#if 0
		testing::InitGoogleMock(&argc, argv);
		//testing::InitGoogleTest(&argc, argv);  // Do not need to be called.
#else
		testing::InitGoogleTest(&argc, argv);
#endif
		retval = RUN_ALL_TESTS();
	}
	catch (const std::bad_alloc &e)
	{
		std::cout << "std::bad_alloc caught: " << e.what() << std::endl;
		retval = EXIT_FAILURE;
	}
	catch (const std::exception &e)
	{
		std::cout << "std::exception caught: " << e.what() << std::endl;
		retval = EXIT_FAILURE;
	}
	catch (...)
	{
		std::cout << "Unknown exception caught" << std::endl;
		retval = EXIT_FAILURE;
	}

	std::cout << "Press any key to exit ..." << std::endl;
	std::cin.get();

	return retval;
}

//-----------------------------------------------------------------------------
// CppUnit.

#elif defined(__SWL_UNIT_TEST__USE_CPP_UNIT)

int main(int argc, char *argv[])
{
	int retval = EXIT_SUCCESS;
	try
	{
		CppUnit::TextUi::TestRunner runner;
		runner.addTest(CppUnit::TestFactoryRegistry::getRegistry().makeTest());

		//runner.setOutputter();
		runner.run();
	}
    catch (const std::bad_alloc &e)
	{
		std::cout << "std::bad_alloc caught: " << e.what() << std::endl;
		retval = EXIT_FAILURE;
	}
	catch (const std::exception &e)
	{
		std::cout << "std::exception caught: " << e.what() << std::endl;
		retval = EXIT_FAILURE;
	}
	catch (...)
	{
		std::cout << "Unknown exception caught" << std::endl;
		retval = EXIT_FAILURE;
	}

	std::cout << "Press any key to exit ..." << std::endl;
	std::cin.get();

	return retval;
}

#endif
