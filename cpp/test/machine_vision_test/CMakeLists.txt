if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_machine_vision_test_d)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_machine_vision_test)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

set(SRCS
	main.cpp
)
if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		swl_machine_vision_d
		swl_base_d
	)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		swl_machine_vision
		swl_base
	)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

add_executable(${TARGET} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})

add_test(test/${TARGET} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET})
