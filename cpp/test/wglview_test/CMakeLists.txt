if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_wglview_test_d)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_wglview_test)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

set(SRCS
	ChildFrm.cpp
	MainFrm.cpp
	SceneGraphShape.cpp
	stdafx.cpp
	ViewEventHandler.cpp
	ViewStateMachine.cpp
	WglSceneGraphView.cpp
	WglViewTest.cpp
	WglViewTest.rc
	WglViewTestDoc.cpp
	WglViewTestView.cpp
)
if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		swl_glutil_d
		swl_view_d
		swl_graphics_d
		swl_util_d
		swl_math_d
		swl_base_d
		gdiplus
		${GLUT_LIBRARIES}
		${OPENGL_LIBRARIES}
	)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		swl_glutil
		swl_view
		swl_graphics
		swl_util
		swl_math
		swl_base
		gdiplus
		${GLUT_LIBRARIES}
		${OPENGL_LIBRARIES}
	)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

#add_definitions(-D_AFXDLL)

add_executable(${TARGET} WIN32 ${SRCS})
target_link_libraries(${TARGET} ${LIBS})

add_test(
	NAME unit_test/${TARGET}
	COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}
	WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)
