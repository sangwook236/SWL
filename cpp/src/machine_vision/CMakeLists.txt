if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_machine_vision_d)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_machine_vision)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

set(SRCS
	ImageRectificationUsingOpenCV.cpp
	ImageUndistortionUsingOpenCV.cpp
	KinectSensor.cpp
)
if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		swl_rnd_util_d
		swl_math_d
		swl_base_d
		${OpenCV_LIBS}
	)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		swl_rnd_util
		swl_math
		swl_base
		${OpenCV_LIBS}
	)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

add_definitions(-DSWL_MACHINE_VISION_EXPORT;)

add_library(${TARGET} ${LIB_TYPE} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})
