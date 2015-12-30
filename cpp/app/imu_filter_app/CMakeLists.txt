include_directories(${MY_HW_DEV_EXT_INCLUDE})
link_directories(${MY_HW_DEV_EXT_LIB})

if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_imu_filter_app_d)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_imu_filter_app)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

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
	set(LIBS
		swl_rnd_util_d
		swl_math_d
		adisusbz_d
		${GSL_LIBRARIES}
		${LAPACK_LIBRARIES}
	)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		swl_rnd_util
		swl_math
		adisusbz
		${GSL_LIBRARIES}
		${LAPACK_LIBRARIES}
	)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

add_executable(${TARGET} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})
