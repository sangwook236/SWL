if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_rnd_util_test_d)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_rnd_util_test)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

set(SRCS
	main.cpp
)
if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		swl_rnd_util_d
		swl_math_d
		swl_base_d
		${OpenCV_LIBS}
		${GSL_LIBRARIES}
		${LAPACK_LIBRARIES}
	)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		swl_rnd_util
		swl_math
		swl_base
		${OpenCV_LIBS}
		${GSL_LIBRARIES}
		${LAPACK_LIBRARIES}
	)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

add_executable(${TARGET} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})

add_test(test/${TARGET} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET})
