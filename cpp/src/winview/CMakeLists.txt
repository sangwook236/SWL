if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_winview_d)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_winview)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

set(SRCS                                 
	GdiBitmapBufferedContext.cpp
	GdiContext.cpp
	GdiContextBase.cpp
	GdiplusBitmapBufferedContext.cpp
	GdiplusContext.cpp
	GdiplusContextBase.cpp
	GdiPrintContext.cpp
	GdiRubberBand.cpp
	WglBitmapBufferedContext.cpp
	WglContextBase.cpp
	WglDoubleBufferedContext.cpp
	WglFont.cpp
	WglPrintContext.cpp
	WglViewBase.cpp
	WglViewCaptureApi.cpp
	WglViewPrintApi.cpp
	WinClipboardApi.cpp
	WinViewBase.cpp
	WinViewCaptureApi.cpp
	WinViewPrintApi.cpp
)                                        
if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		swl_graphics_d
		swl_glutil_d
		swl_view_d
		swl_util_d
		swl_math_d
		swl_base_d
		gdiplus
		${GLUT_LIBRARIES}
		${OPENGL_LIBRARIES}
	)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		swl_graphics
		swl_glutil
		swl_view
		swl_util
		swl_math
		swl_base
		gdiplus
		${GLUT_LIBRARIES}
		${OPENGL_LIBRARIES}
	)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

add_definitions(-DSWL_WIN_VIEW_EXPORT)

add_library(${TARGET} ${LIB_TYPE} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})
