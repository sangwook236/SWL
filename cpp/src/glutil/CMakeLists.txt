if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_glutil_d)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_glutil)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

set(SRCS
	GLCamera.cpp
	GLCreateDisplayListVisitor.cpp
	GLDisplayListCallableInterface.cpp
	GLPickObjectVisitor.cpp
	GLPrintSceneVisitor.cpp
	GLRenderSceneVisitor.cpp
	GLShape.cpp
)
if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		swl_view_d
		swl_graphics_d
		swl_math_d
		swl_base_d
		${GLUT_LIBRARIES}
		${OPENGL_LIBRARIES}
	)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		swl_view
		swl_graphics
		swl_math
		swl_base
		${GLUT_LIBRARIES}
		${OPENGL_LIBRARIES}
	)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

add_definitions(-DSWL_GL_UTIL_EXPORT)

if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
	add_compile_options(-fPIC)
endif(CMAKE_CXX_COMPILER_ID MATCHES GNU)

add_library(${TARGET} ${LIB_TYPE} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})
