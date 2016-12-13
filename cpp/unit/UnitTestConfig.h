#if !defined(__SWL_UNIT_TEST__UNIT_TEST_CONFIG__H_)
#define __SWL_UNIT_TEST__UNIT_TEST_CONFIG__H_ 1


#if !defined(__SWL_UNIT_TEST__USE_BOOST_TEST) && !defined(__SWL_UNIT_TEST__USE_GOOGLE_TEST) && !defined(__SWL_UNIT_TEST__USE_CPP_UNIT)
//#define __SWL_UNIT_TEST__USE_BOOST_TEST 1
#define __SWL_UNIT_TEST__USE_GOOGLE_TEST 1
//#define __SWL_UNIT_TEST__USE_CPP_UNIT 1  // Deprecated.
#endif


//-----------------------------------------------------------------------------
// Boost Test.

#if defined(__SWL_UNIT_TEST__USE_BOOST_TEST)

//#define BOOST_TEST_MODULE SWL.UnitTest
#include <boost/test/unit_test.hpp>
#include <boost/bind.hpp>

//-----------------------------------------------------------------------------
// Google Test.

#elif defined(__SWL_UNIT_TEST__USE_GOOGLE_TEST)

//#include <gmock/gmock.h>
#include <gtest/gtest.h>

// Automatic linking.
#if defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64)
#	if defined(_DEBUG)
#		pragma comment(lib, "gtestd.lib")
//#		pragma comment(lib, "gtest_maind.lib")
#	else
#		pragma comment(lib, "gtest.lib")
//#		pragma comment(lib, "gtest_main.lib")
#	endif
#endif

//-----------------------------------------------------------------------------
// CppUnit.

#elif defined(__SWL_UNIT_TEST__USE_CPP_UNIT)

#include <cppunit/extensions/HelperMacros.h>

// Automatic linking.
#if defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64)
#	if defined(_DEBUG)
#		pragma comment(lib, "cppunitd_dll.lib")
#		if defined(_UNICODE) || defined(UNICODE)
#			pragma comment(lib, "testrunnerud.lib")
#		else
#			pragma comment(lib, "testrunnerd.lib")
#		endif
#	else
#		pragma comment(lib, "cppunit_dll.lib")
#		if defined(_UNICODE) || defined(UNICODE)
#			pragma comment(lib, "testrunneru.lib")
#		else
#			pragma comment(lib, "testrunner.lib")
#		endif
#	endif
#endif

//-----------------------------------------------------------------------------
// Otherwise

#else

#error Configurations in SWL.UnitTest are incorrect. refer to "${SWL_C++_HOME}/unit/UnitTestConfig.h".

#endif


#endif  // __SWL_UNIT_TEST__UNIT_TEST_CONFIG__H_
