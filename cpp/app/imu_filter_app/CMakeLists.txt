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

	# AdisUsbz library
	adisusbz_lib/AdisUsbz.cpp
)
set(LIBS
	swl_rnd_util
	swl_math
	${GSL_LIBRARIES}
	${LAPACK_LIBRARIES}
)

add_executable(${TARGET} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})
