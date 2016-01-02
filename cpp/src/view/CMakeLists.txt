if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_view_d)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_view)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

set(SRCS
	ViewCamera2.cpp
	ViewCamera3.cpp
	ViewEventController.cpp
	ViewMatrix.cpp
)
if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		swl_math_d
		swl_base_d
	)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		swl_math
		swl_base
	)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

add_definitions(-DSWL_VIEW_EXPORT)

if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
	add_compile_options(-fPIC)
endif(CMAKE_CXX_COMPILER_ID MATCHES GNU)

add_library(${TARGET} ${LIB_TYPE} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})
