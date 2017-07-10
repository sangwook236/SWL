set(TARGET swl_util_test)

set(SRCS
	main.cpp
	WaveDataTest.cpp
)
set(LIBS
	swl_util
	swl_math
	swl_base
)

add_executable(${TARGET} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})

if(NOT USE_UNIT_TESTCASES_ONLY_AS_TESTS)
add_test(
	NAME test/${TARGET}
	COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}
	WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)
endif(NOT USE_UNIT_TESTCASES_ONLY_AS_TESTS)
