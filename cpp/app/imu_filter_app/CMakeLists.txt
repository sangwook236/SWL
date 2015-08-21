include_directories(${MY_HW_DEV_EXT_INCLUDE})
link_directories(${MY_HW_DEV_EXT_LIB})

set(TARGET swl_imu_filter_app)
set(SRCS
	main.cpp
	ImuCalibration.cpp
	ImuExtendedKalmanFilterRunner.cpp
	ImuExtendedKalmanFilterWithCalibration.cpp
	ImuKalmanFilter.cpp
	ImuSystem.cpp
	ImuUnscentedKalmanFilterRunner.cpp
	ImuUnscentedKalmanFilterWithCalibration.cpp
)
if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS swl_rnd_util swl_math adisusbz_d gsl cblas)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS swl_rnd_util swl_math swl_winutil adisusbz gsl cblas)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

#add_definitions(-D_AFXDLL)

add_executable(${TARGET} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})
