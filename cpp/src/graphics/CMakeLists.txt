set(TARGET swl_graphics)

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
set(LIBS
	swl_math
	swl_base
)

add_definitions(-DSWL_GRAPHICS_EXPORT)

if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
	add_compile_options(-fPIC)
endif(CMAKE_CXX_COMPILER_ID MATCHES GNU)

add_library(${TARGET} ${LIB_TYPE} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})
