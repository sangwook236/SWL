if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_kinematics_test_d)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_kinematics_test)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

set(SRCS
	main.cpp
)
if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		swl_kinematics_d
		swl_math_d
		swl_base_d
	)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		swl_kinematics
		swl_math
		swl_base
	)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

add_executable(${TARGET} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})

add_test(test/${TARGET} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET})
