include_directories(${MY_HW_DEV_EXT_INCLUDE})
link_directories(${MY_HW_DEV_EXT_LIB})

if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_gps_aided_imu_filter_app_d)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_gps_aided_imu_filter_app)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

set(SRCS
	stdafx.cpp
	Adis16350Interface.cpp
	DataDefinition.cpp
	FileBasedFilteringDialog.cpp
	GpsAidedImuFilterDialog.cpp
	GpsAidedImuFilterRunner.cpp
	GpsAidedImuSystem.cpp
	GpsInterface.cpp
	swl_gps_aided_imu_filter_app.cpp
	swl_gps_aided_imu_filter_app.rc
	swl_gps_aided_imu_filter_appDlg.cpp
)
if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		swl_winutil_d
		swl_rnd_util_d
		swl_math nmea_d
		adisusbz_d
		${GSL_LIBRARIES}
		${LAPACK_LIBRARIES}
	)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		swl_winutil
		swl_rnd_util
		swl_math nmea
		adisusbz
		${GSL_LIBRARIES}
		${LAPACK_LIBRARIES}
	)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

add_definitions(-D_AFXDLL)

add_executable(${TARGET} WIN32 ${SRCS})
target_link_libraries(${TARGET} ${LIBS})
