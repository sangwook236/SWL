set(TARGET swl_unit_test)

string(REGEX REPLACE "([^;]+)" "../\\1" TESTCASE_SRCS "${TESTCASE_SRCS}")

set(SRCS
	TextUiUnitTestMain.cpp
	${TESTCASE_SRCS}
)

if(GTEST_FOUND)
	add_definitions(-D__SWL_UNIT_TEST__USE_GOOGLE_TEST)
	set(TEST_LIBS
		${GTEST_BOTH_LIBRARIES}
	)
elseif(CPPUNIT_FOUND)
	add_definitions(-D__SWL_UNIT_TEST__USE_CPP_UNIT)  # deprecated.
	set(TEST_LIBS
		$[CPPUNIT_LIBRARIES}
	)
else()
	add_definitions(-D__SWL_UNIT_TEST__USE_BOOST_TEST)
	set(TEST_LIBS
		${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
	)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
	add_compile_options(-std=c++11)
endif(CMAKE_CXX_COMPILER_ID MATCHES GNU)

add_executable(${TARGET} ${SRCS})
target_link_libraries(${TARGET} ${LIBS} ${TEST_LIBS})

add_test(
	NAME unit_test/${TARGET}
	COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}
	WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)
