set(TARGET swl_machine_vision)

set(SRCS
	ImageRectificationUsingOpenCV.cpp
	ImageUndistortionUsingOpenCV.cpp
	DerivativesOfGaussian.cpp
	KinectSensor.cpp
	ScaleSpace.cpp
)
set(LIBS
	swl_rnd_util
	swl_math
	swl_base
	${OpenCV2_LIBRARIES}
)

add_definitions(-DSWL_MACHINE_VISION_EXPORT;)

if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
	add_compile_options(-fPIC)
endif(CMAKE_CXX_COMPILER_ID MATCHES GNU)

add_library(${TARGET} ${LIB_TYPE} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})
