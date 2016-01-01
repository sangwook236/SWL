if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_cppunit_mfc_d)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_cppunit_mfc)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

set(SRCS
	stdafx.cpp
	UnitTestMfc.cpp
	UnitTestMfc.rc
	UnitTestMfcDlg.cpp
	${BASE_SRCS}
	${MATH_SRCS}
	${UTIL_SRCS}
	${RND_UTIL_SRCS}
)

#add_definitions(-D__SWL_UNIT_TEST__USE_BOOST_TEST)
#add_definitions(-D__SWL_UNIT_TEST__USE_GOOGLE_TEST)
add_definitions(-D__SWL_UNIT_TEST__USE_CPP_UNIT)  // deprecated.
#add_definitions(-D_AFXDLL)

add_executable(${TARGET} WIN32 ${SRCS})
target_link_libraries(${TARGET} ${LIBS})

add_test(
	NAME unit_test/${TARGET}
	COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}
	WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)
