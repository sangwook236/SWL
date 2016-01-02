if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_graphics_d)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_graphics)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

set(SRCS
	Appearance.cpp
	CoordinateFrame.cpp
	Geometry.cpp
	GeometryPoolMgr.cpp
	GraphicsObj.cpp
	ObjectPickerMgr.cpp
	PickableInterface.cpp
	Shape.cpp
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

add_definitions(-DSWL_GRAPHICS_EXPORT)

if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
	add_compile_options(-fPIC)
endif(CMAKE_CXX_COMPILER_ID MATCHES GNU)

add_library(${TARGET} ${LIB_TYPE} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})
