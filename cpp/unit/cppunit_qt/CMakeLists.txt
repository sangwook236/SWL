set(TARGET swl_cppunit_qt)

set(SRCS
	main.cpp
	${TESTCASE_SRCS}
)

#add_definitions(-D__SWL_UNIT_TEST__USE_BOOST_TEST)
#add_definitions(-D__SWL_UNIT_TEST__USE_GOOGLE_TEST)
add_definitions(-D__SWL_UNIT_TEST__USE_CPP_UNIT)  // deprecated.

if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
	add_compile_options(-std=c++11)
endif(CMAKE_CXX_COMPILER_ID MATCHES GNU)

add_executable(${TARGET} ${SRCS})
#target_link_libraries(${TARGET} ${LIBS} QtTestRunner QtGui4 QtCore4)
#target_link_libraries(${TARGET} ${LIBS} QtTestRunner ${QT_LIBRARIES})
target_link_libraries(${TARGET} ${LIBS} QtTestRunner Qt4::QtGui Qt4::QtCore)

add_test(
	NAME unit_test/${TARGET}
	COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}
	WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)
