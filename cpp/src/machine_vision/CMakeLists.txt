set(TARGET swl_machine_vision)

set(SRCS
	BoundaryExtraction.cpp
	DerivativesOfGaussian.cpp
	ImageFilter.cpp
	ImageRectificationUsingOpenCV.cpp
	ImageUndistortionUsingOpenCV.cpp
	KinectSensor.cpp
	ScaleSpace.cpp
	NonMaximaSuppression.cpp
	SkeletonAlgorithm.cpp
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
