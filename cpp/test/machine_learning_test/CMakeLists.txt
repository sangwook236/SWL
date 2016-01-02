if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_machine_learning_test_d)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_machine_learning_test)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

set(SRCS
	main.cpp
	CliffWalkingSystem.cpp
	td_learning.cpp
	WindyGridWorldSystem.cpp
)
if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		swl_machine_learning_d
		swl_posixutil_d
		swl_math_d
		swl_base_d
	)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		swl_machine_learning
		swl_posixutil
		swl_math
		swl_base
	)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
	add_compile_options(-fpermissive)
endif(CMAKE_CXX_COMPILER_ID MATCHES GNU)

add_executable(${TARGET} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})

add_test(test/${TARGET} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET})
