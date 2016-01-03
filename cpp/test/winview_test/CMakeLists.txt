if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_winview_test_d)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_winview_test)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

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
if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		swl_winview_d
		swl_view_d
		swl_base_d
		gdiplus
	)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		swl_winview
		swl_view
		swl_base
		gdiplus
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
