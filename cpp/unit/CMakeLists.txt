#add_subdirectory(text_ui)
#add_subdirectory(testcase)

set(TEXT_UI_SRCS text_ui/TextUiUnitTestMain.cpp)
set(BASE_SRCS testcase/base/LoggerTest.cpp)
set(MATH_SRCS testcase/math/BaseFieldConversionTest.cpp testcase/math/LineSegmentTest.cpp testcase/math/LineTest.cpp testcase/math/PlaneTest.cpp testcase/math/TriangleTest.cpp)
set(UTIL_SRCS testcase/util/EchoTcpSocketConnection.cpp testcase/util/EchoTcpSocketServerTest.cpp testcase/util/EchoTcpSocketSession.cpp testcase/util/EchoTcpSocketTest.cpp testcase/util/LineROITest.cpp testcase/util/PolygonROITest.cpp testcase/util/PolylineROITest.cpp testcase/util/RectangleROITest.cpp testcase/util/RegionOfInterestMgrTest.cpp testcase/util/RegionOfInterestTest.cpp testcase/util/ROIWithVariablePointsTest.cpp)

set(TARGET swl_unit_test)
set(SRCS ${TEXT_UI_SRCS} ${BASE_SRCS} ${MATH_SRCS} ${UTIL_SRCS})
if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(
		LIBS
#		swl_winview swl_winutil swl_view.lib swl_util.lib swl_glutil.lib swl_math.lib swl_base.lib log4cxx.lib
		swl_util.lib swl_math.lib swl_base log4cxx.lib
	)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(
		LIBS
#		swl_winview swl_winutil swl_view.lib swl_util.lib swl_glutil.lib swl_math.lib swl_base.lib log4cxx.lib
		swl_util.lib swl_math.lib swl_base log4cxx.lib
	)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

add_definitions(-D__SWL_UNIT_TEST__USE_BOOST_UNIT)

add_executable(${TARGET} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})

add_test(
	NAME unit_test/${TARGET}
	COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}
	WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)
