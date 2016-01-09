set(TARGET swl_gesture_recognition_test)

set(SRCS
	main.cpp
	GestureClassifierBasedOnTHoGTest.cpp
	GestureClassifierByHistogramTest.cpp
	HistogramGenerator.cpp
)
set(LIBS
	swl_pattern_recognition
	swl_rnd_util
	swl_base
	${OpenCV2_LIBRARIES}
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
