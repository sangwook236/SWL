set(TARGET swl_glutil)

set(SRCS
	GLCamera.cpp
	GLCreateDisplayListVisitor.cpp
	GLDisplayListCallableInterface.cpp
	GLPickObjectVisitor.cpp
	GLPrintSceneVisitor.cpp
	GLRenderSceneVisitor.cpp
	GLShape.cpp
)
set(LIBS
	swl_view
	swl_graphics
	swl_math
	swl_base
	${GLUT_LIBRARIES}
	${OPENGL_LIBRARIES}
)

add_definitions(-DSWL_GL_UTIL_EXPORT)

if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
	add_compile_options(-fPIC)
endif(CMAKE_CXX_COMPILER_ID MATCHES GNU)

add_library(${TARGET} ${LIB_TYPE} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})
