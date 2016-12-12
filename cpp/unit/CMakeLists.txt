set(TESTCASE_SRCS
	# testcase/base
	testcase/base/LogTest.cpp

	# testcase/math
	testcase/math/BaseFieldConversionTest.cpp
	testcase/math/CurveFittingTest.cpp
	testcase/math/DataNormalizationTest.cpp
	testcase/math/GeometryUtilTest.cpp
	testcase/math/LineSegmentTest.cpp
	testcase/math/LineTest.cpp
	testcase/math/MathUtilTest.cpp
	testcase/math/PlaneTest.cpp
	testcase/math/StatisticTest.cpp
	testcase/math/TriangleTest.cpp

	# testcase/util
	testcase/util/EchoTcpSocketConnection.cpp
#	testcase/util/EchoTcpSocketServerTest.cpp
	testcase/util/EchoTcpSocketSession.cpp
#	testcase/util/EchoTcpSocketTest.cpp
	testcase/util/LineROITest.cpp
	testcase/util/PolygonROITest.cpp
	testcase/util/PolylineROITest.cpp
	testcase/util/RectangleROITest.cpp
	testcase/util/RegionOfInterestMgrTest.cpp
	testcase/util/RegionOfInterestTest.cpp
	testcase/util/ROIWithVariablePointsTest.cpp

	# testcase/rnd_util
	testcase/rnd_util/SortTest.cpp
)

set(PLATFORM_INDEPENDENT_LIBS
	swl_machine_vision
	swl_pattern_recognition
	swl_machine_learning
	swl_kinematics
	swl_rnd_util
	swl_glutil
	swl_view
	swl_graphics
	swl_util
	swl_math
	swl_base
	${LOG4CXX_LIBRARIES}
	${Boost_SYSTEM_LIBRARY}
	${CMAKE_THREAD_LIBS_INIT}
)
if(WIN32 AND NOT CYGWIN)
	set(PLATFORM_DEPENDENT_LIBS
		swl_winview
		swl_winutil
	)
else(WIN32 AND NOT CYGWIN)
	set(PLATFORM_DEPENDENT_LIBS
		swl_posixutil
	)
endif(WIN32 AND NOT CYGWIN)
set(LIBS
	${PLATFORM_DEPENDENT_LIBS}
	${PLATFORM_INDEPENDENT_LIBS}
)

add_subdirectory(text_ui)
if(WIN32 AND NOT CYGWIN)
	add_subdirectory(cppunit_mfc)
endif(WIN32 AND NOT CYGWIN)
if(QT_FOUND)
#	add_subdirectory(cppunit_qt)
endif(QT_FOUND)
