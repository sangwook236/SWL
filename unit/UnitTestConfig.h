#if !defined(__SWL_UNIT_TEST__UNIT_TEST_CONFIG__H_)
#define __SWL_UNIT_TEST__UNIT_TEST_CONFIG__H_ 1


//#define __SWL_UNIT_TEST__USE_BOOST_UNIT 1
#define __SWL_UNIT_TEST__USE_CPP_UNIT 1


#if defined(__SWL_UNIT_TEST__USE_BOOST_UNIT)

//#define BOOST_TEST_MODULE SWL.UnitTest
#include <boost/test/unit_test.hpp>
#include <boost/bind.hpp>

#elif defined(__SWL_UNIT_TEST__USE_CPP_UNIT)

#include <cppunit/extensions/HelperMacros.h>
#if defined(_DEBUG)
#	pragma comment(lib, "cppunitd_dll.lib")
#	if defined(_UNICODE) || defined(UNICODE)
#		pragma comment(lib, "testrunnerud.lib")
#	else
#		pragma comment(lib, "testrunnerd.lib")
#	endif
#else
#	pragma comment(lib, "cppunit_dll.lib")
#	if defined(_UNICODE) || defined(UNICODE)
#		pragma comment(lib, "testrunneru.lib")
#	else
#		pragma comment(lib, "testrunner.lib")
#	endif
#endif

#else

#error configurations in SWL.UnitTest are incorrect. refer to "${swl_root}/unit/UnitTestConfig.h".

#endif


#endif  // __SWL_UNIT_TEST__UNIT_TEST_CONFIG__H_
