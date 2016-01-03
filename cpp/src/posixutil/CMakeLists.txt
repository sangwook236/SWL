set(TARGET swl_posixutil)

set(SRCS
	PosixTimer.cpp
)
set(LIBS
	swl_base
)

add_definitions(-DSWL_POSIX_UTIL_EXPORT)

if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
	add_compile_options(-fPIC)
endif(CMAKE_CXX_COMPILER_ID MATCHES GNU)

add_library(${TARGET} ${LIB_TYPE} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})
