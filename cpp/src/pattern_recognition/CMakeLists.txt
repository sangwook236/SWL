set(TARGET swl_pattern_recognition)

set(SRCS
	GestureClassifierByHistogram.cpp
	GestureType.cpp
	HistogramGenerator.cpp
	MotionSegmenter.cpp
)
set(LIBS
	swl_rnd_util
	swl_base
	${OpenCV_LIBRARIES}
)

add_definitions(-DSWL_PATTERN_RECOGNITION_EXPORT)

if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
	add_compile_options(-fPIC)
endif(CMAKE_CXX_COMPILER_ID MATCHES GNU)

add_library(${TARGET} ${LIB_TYPE} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})
