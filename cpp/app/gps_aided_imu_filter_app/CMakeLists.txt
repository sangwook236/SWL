include_directories(${MY_HW_DEV_EXT_INCLUDE})
link_directories(${MY_HW_DEV_EXT_LIB})

set(TARGET swl_gps_aided_imu_filter_app)

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
set(LIBS
	swl_winutil
	swl_rnd_util
	swl_math
	nmea
	adisusbz
	${GSL_LIBRARIES}
	${LAPACK_LIBRARIES}
)

add_definitions(-D_AFXDLL)

add_executable(${TARGET} WIN32 ${SRCS})
target_link_libraries(${TARGET} ${LIBS})
