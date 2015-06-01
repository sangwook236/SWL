include_directories(${SW_DEV_HW_EXT_INCLUDE})
link_directories(${SW_DEV_HW_EXT_LIB})

set(TARGET swl_gps_aided_imu_filter_app)
set(SRCS stdafx.cpp Adis16350Interface.cpp DataDefinition.cpp FileBasedFilteringDialog.cpp GpsAidedImuFilterDialog.cpp GpsAidedImuFilterRunner.cpp GpsAidedImuSystem.cpp GpsInterface.cpp swl_gps_aided_imu_filter_app.cpp swl_gps_aided_imu_filter_appDlg.cpp swl_gps_aided_imu_filter_app.rc)
if(CMAKE_BUILD_TYPE MATCHES DEBUG)
    set(LIBS swl_winutil swl_rnd_util swl_math nmea_d adisusbz_d gsl cblas)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
    set(LIBS swl_winutil swl_rnd_util swl_math nmea adisusbz gsl cblas)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

add_definitions(-D_AFXDLL)

add_executable(${TARGET} WIN32 ${SRCS})
target_link_libraries(${TARGET} ${LIBS})
