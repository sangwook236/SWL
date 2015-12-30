if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_gesture_recognition_test_d)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_gesture_recognition_test)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

set(SRCS
	main.cpp
	GestureClassifierBasedOnTHoGTest.cpp
	GestureClassifierByHistogramTest.cpp
	HistogramGenerator.cpp
)
if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		swl_pattern_recognition_d
		swl_rnd_util_d
		swl_base_d
		${OpenCV_LIBS}
	)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		swl_pattern_recognition
		swl_rnd_util
		swl_base
		${OpenCV_LIBS}
	)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

add_executable(${TARGET} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})

add_test(test/${TARGET} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET})
