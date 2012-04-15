#if defined(WIN32)
#include <vld/vld.h>
#endif
#include "swl/Config.h"
#include "../UnitTestConfig.h"

#if defined(__SWL_UNIT_TEST__USE_BOOST_UNIT)
#include <boost/test/detail/unit_test_parameters.hpp>
#include <fstream>
#elif defined(__SWL_UNIT_TEST__USE_CPP_UNIT)
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/ui/text/TestRunner.h>
#endif
#include <iostream>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


#if defined(__SWL_UNIT_TEST__USE_BOOST_UNIT)

namespace {
namespace local {

//
//	runtime configration: command line arguments
//		--show_progress=yes --log_level=message --run_test=*
//

struct BoostUnitGlobalFixture
{
    BoostUnitGlobalFixture()
	: logStream_("swl_unittest.log")
	{
		boost::unit_test::unit_test_log.set_stream(logStream_);
		boost::unit_test::unit_test_log.set_format(boost::unit_test::XML);
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

#elif defined(__SWL_UNIT_TEST__USE_CPP_UNIT)

int main(int argc, char *argv[])
{
	try
	{
		CppUnit::TextUi::TestRunner runner;
		runner.addTest(CppUnit::TestFactoryRegistry::getRegistry().makeTest());

		//runner.setOutputter();
		runner.run();
	}
	catch (const std::exception &e)
	{
		std::cout << "std::exception caught: " << e.what() << std::endl;
		return -1;
	}
	catch (...)
	{
		std::cout << "unknown exception caught" << std::endl;
		return -1;
	}

	std::cout << "press any key to exit ..." << std::endl;
	std::cin.get();

	return 0;
}

#endif
