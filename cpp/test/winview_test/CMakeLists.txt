set(TARGET swl_winview_test)

set(SRCS
	ChildFrm.cpp
	MainFrm.cpp
	stdafx.cpp
	ViewEventHandler.cpp
	ViewStateMachine.cpp
	WinViewTest.cpp
	WinViewTest.rc
	WinViewTestDoc.cpp
	WinViewTestView.cpp
)
set(LIBS
	swl_winview
	swl_view
	swl_base
	gdiplus
)

#add_definitions(-D_AFXDLL)

add_executable(${TARGET} WIN32 ${SRCS})
target_link_libraries(${TARGET} ${LIBS})

add_test(
	NAME unit_test/${TARGET}
	COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}
	WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)
