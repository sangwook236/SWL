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
#		${OpenCV_LIBS}
		opencv_video
		opencv_imgproc
		opencv_highgui
		opencv_core
	)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		swl_pattern_recognition
		swl_rnd_util
		swl_base
#		${OpenCV_LIBS}
		opencv_video
		opencv_imgproc
		opencv_highgui
		opencv_core
	)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

add_executable(${TARGET} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})

add_test(
	NAME unit_test/${TARGET}
	COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}
	WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)
