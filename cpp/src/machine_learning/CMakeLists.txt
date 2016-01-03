set(TARGET swl_machine_learning)

set(SRCS
	TDLearningBase.cpp
)
set(LIBS
	swl_rnd_util
	swl_math
	swl_base
)

add_definitions(-DSWL_MACHINE_LEARNING_EXPORT)

if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
	add_compile_options(-fPIC)
endif(CMAKE_CXX_COMPILER_ID MATCHES GNU)

add_library(${TARGET} ${LIB_TYPE} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})
