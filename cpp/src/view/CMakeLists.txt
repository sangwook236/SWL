set(TARGET swl_view)

set(SRCS
	ViewCamera2.cpp
	ViewCamera3.cpp
	ViewEventController.cpp
	ViewMatrix.cpp
)
set(LIBS
	swl_math
	swl_base
)

add_definitions(-DSWL_VIEW_EXPORT)

if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
	add_compile_options(-fPIC)
endif(CMAKE_CXX_COMPILER_ID MATCHES GNU)

add_library(${TARGET} ${LIB_TYPE} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})
