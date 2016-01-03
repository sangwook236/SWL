set(TARGET swl_machine_learning_test)

set(SRCS
	main.cpp
	CliffWalkingSystem.cpp
	td_learning.cpp
	WindyGridWorldSystem.cpp
)
set(PLATFORM_INDEPENDENT_LIBS
	swl_machine_learning
	swl_math
	swl_base
)
if(WIN32 AND NOT CYGWIN)
	set(PLATFORM_DEPENDENT_LIBS
		swl_winutil
	)
else(WIN32 AND NOT CYGWIN)
	set(PLATFORM_DEPENDENT_LIBS
		swl_posixutil
	)
endif(WIN32 AND NOT CYGWIN)
set(LIBS
	${PLATFORM_DEPENDENT_LIBS}
	${PLATFORM_INDEPENDENT_LIBS}
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
