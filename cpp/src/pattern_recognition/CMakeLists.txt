if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_pattern_recognition_d)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_pattern_recognition)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

set(SRCS
	INotifier.cpp
	IObserver.cpp
	LogException.cpp
	ReturnException.cpp
	String.cpp
)
if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		swl_rnd_util_d
		swl_base_d
		${OpenCV_LIBS}
	)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		swl_rnd_util
		swl_base
		${OpenCV_LIBS}
	)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

add_definitions(-DSWL_PATTERN_RECOGNITION_EXPORT)

add_library(${TARGET} ${LIB_TYPE} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})
