set(TARGET swl_wglview_test)

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
set(LIBS
	swl_winview
	swl_view
	swl_glutil
	swl_graphics
	swl_util
	swl_math
	swl_base
	gdiplus
	${GLUT_LIBRARIES}
	${OPENGL_LIBRARIES}
)

#add_definitions(-D_AFXDLL)

add_executable(${TARGET} WIN32 ${SRCS})
target_link_libraries(${TARGET} ${LIBS})

if(NOT USE_UNIT_TESTCASES_ONLY_AS_TESTS)
	add_test(
		NAME test/${TARGET}
		COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}
		WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
	)
endif(NOT USE_UNIT_TESTCASES_ONLY_AS_TESTS)
