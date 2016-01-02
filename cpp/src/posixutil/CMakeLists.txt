if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_posixutil_d)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_posixutil)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

set(SRCS
	PosixTimer.cpp
)
if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		swl_base_d
	)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		swl_base
	)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

add_definitions(-DSWL_POSIX_UTIL_EXPORT)

if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
	add_compile_options(-fPIC)
endif(CMAKE_CXX_COMPILER_ID MATCHES GNU)

add_library(${TARGET} ${LIB_TYPE} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})
