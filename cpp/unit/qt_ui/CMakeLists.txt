if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_unit_test_qt_d)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_unit_test_qt)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

set(SRCS
	main.cpp
	${BASE_SRCS}
	${MATH_SRCS}
	${UTIL_SRCS}
	${RND_UTIL_SRCS}
)

#add_definitions(-D__SWL_UNIT_TEST__USE_BOOST_UNIT)
add_definitions(-D__SWL_UNIT_TEST__USE_CPP_UNIT)

add_executable(${TARGET} ${SRCS})
if(CMAKE_BUILD_TYPE MATCHES DEBUG)
#	target_link_libraries(${TARGET} QtGuid4 QtCored4)
	target_link_libraries(${TARGET} ${LIBS} Qt4::QtGui Qt4::QtCore)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
#	target_link_libraries(${TARGET} ${LIBS} QtGui4 QtCore4)
	target_link_libraries(${TARGET} ${LIBS} Qt4::QtGui Qt4::QtCore)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

add_test(
	NAME unit_test/${TARGET}
	COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}
	WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)
