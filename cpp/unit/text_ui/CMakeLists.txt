if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_unit_test_d)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_unit_test)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

set(SRCS
	TextUiUnitTestMain.cpp
	${BASE_SRCS}
	${MATH_SRCS}
	${UTIL_SRCS}
	${RND_UTIL_SRCS}
)

add_definitions(-D__SWL_UNIT_TEST__USE_BOOST_UNIT)
#add_definitions(-D__SWL_UNIT_TEST__USE_CPP_UNIT)

add_executable(${TARGET} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})

add_test(
	NAME unit_test/${TARGET}
	COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}
	WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)
