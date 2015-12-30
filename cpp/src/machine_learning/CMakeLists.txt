if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_machine_learning_d)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_machine_learning)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

set(SRCS
	TDLearningBase.cpp
)
if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		swl_rnd_util_d
		swl_math_d
		swl_base_d
	)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		swl_rnd_util
		swl_math
		swl_base
	)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

add_definitions(-DSWL_MACHINE_LEARNING_EXPORT)

add_library(${TARGET} ${LIB_TYPE} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})
