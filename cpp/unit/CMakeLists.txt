set(BASE_SRCS
	testcase/base/LoggerTest.cpp
)
set(MATH_SRCS
	testcase/math/BaseFieldConversionTest.cpp
	testcase/math/LineSegmentTest.cpp
	testcase/math/LineTest.cpp
	testcase/math/PlaneTest.cpp
	testcase/math/TriangleTest.cpp
)
set(UTIL_SRCS
	testcase/util/EchoTcpSocketConnection.cpp
	testcase/util/EchoTcpSocketServerTest.cpp
	testcase/util/EchoTcpSocketSession.cpp
	testcase/util/EchoTcpSocketTest.cpp
	testcase/util/LineROITest.cpp
	testcase/util/PolygonROITest.cpp
	testcase/util/PolylineROITest.cpp
	testcase/util/RectangleROITest.cpp
	testcase/util/RegionOfInterestMgrTest.cpp
	testcase/util/RegionOfInterestTest.cpp
	testcase/util/ROIWithVariablePointsTest.cpp
)
set(RND_UTIL_SRCS
	testcase/rnd_util/SortTest.cpp
)

if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		swl_machine_vision_d
		swl_pattern_recognition_d
		swl_machine_learning_d
		swl_kinematics_d
		swl_rnd_util_d
if(WIN32 AND NOT CYGWIN)
		swl_winview_d
		swl_winutil_d
else(WIN32 AND NOT CYGWIN)
		swl_posixutil_d
endif(WIN32 AND NOT CYGWIN)
		swl_glutil_d
		swl_view_d
		swl_graphics_d
		swl_util_d
		swl_math_d
		swl_base_d
		${LOG4CXX_LIBRARIES}
	)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		swl_machine_vision
		swl_pattern_recognition
		swl_machine_learning
		swl_kinematics
		swl_rnd_util
if(WIN32 AND NOT CYGWIN)
		swl_winview
		swl_winutil
else(WIN32 AND NOT CYGWIN)
		swl_posixutil
endif(WIN32 AND NOT CYGWIN)
		swl_glutil
		swl_view
		swl_graphics
		swl_util
		swl_math
		swl_base
		${LOG4CXX_LIBRARIES}
	)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

add_subdirectory(text_ui)
if(WIN32 AND NOT CYGWIN)
	add_subdirectory(cppunit_mfc)
endif(WIN32 AND NOT CYGWIN)
if(QT_FOUND)
#	add_subdirectory(cppunit_qt)
endif(QT_FOUND)
