#include "swl/Config.h"
#include "UnitTestConfig.h"

#if defined(__SWL_UNIT_TEST__USE_BOOST_UNIT)
#include <boost/test/detail/unit_test_parameters.hpp>
#include <fstream>
#elif defined(__SWL_UNIT_TEST__USE_CPP_UNIT)
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/ui/text/TestRunner.h>
#endif
#include <iostream>


#if defined(_DEBUG)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


#if defined(__SWL_UNIT_TEST__USE_BOOST_UNIT)

namespace {

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

}  // unnamed namespace

//BOOST_GLOBAL_FIXTURE(BoostUnitGlobalFixture);

#if !defined(BOOST_TEST_MODULE)
boost::unit_test_framework::test_suite * init_unit_test_suite(int, char *[])
{
	//boost::unit_test::framework::master_test_suite().add();

	return 0;
}
#endif

#elif defined(__SWL_UNIT_TEST__USE_CPP_UNIT)

int main()
{
	CppUnit::TextUi::TestRunner runner;
	runner.addTest(CppUnit::TestFactoryRegistry::getRegistry().makeTest());

	//runner.setOutputter();
	runner.run();

	std::cout.flush();
	std::cin.get();
	return 0;
}

#endif
