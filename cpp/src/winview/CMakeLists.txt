set(TARGET swl_winview)

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

add_definitions(-DSWL_WIN_VIEW_EXPORT)

add_library(${TARGET} ${LIB_TYPE} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})
