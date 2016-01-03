if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_rnd_util_test_d)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_rnd_util_test)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

set(SRCS
	main.cpp
	ArHmmWithUnivariateNormalMixtureObservationsTest.cpp
	ArHmmWithUnivariateNormalObservationsTest.cpp
	DynamicTimeWarpingTest.cpp
	ExtendedKalmanFilterTest.cpp
	HmmSegmentationTest.cpp
	HmmWithMultinomialObservationsTest.cpp
	HmmWithUnivariateNormalMixtureObservationsTest.cpp
	HmmWithUnivariateNormalObservationsTest.cpp
	HmmWithVonMisesMixtureObservationsTest.cpp
	HmmWithVonMisesObservationsTest.cpp
	HoughTransformTest.cpp
	KalmanFilterTest.cpp
	LevenshteinDistanceTest.cpp
	MetropolisHastingsAlgorithmTest.cpp
	MultivariateNormalMixtureModelTest.cpp
#	ParticleFilterTest.cpp
	RansacTest.cpp
	RejectionSamplingTest.cpp
	SamplingImportanceResamplingTest.cpp
	UnivariateNormalMixtureModelTest.cpp
	UnscentedKalmanFilterTest.cpp
	UnscentedKalmanFilterWithAdditiveNoiseTest.cpp
	VonMisesMixtureModelTest.cpp
)
if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		swl_rnd_util_d
		swl_math_d
		swl_base_d
#		${OpenCV_LIBS}
		opencv_imgproc
		opencv_highgui
		opencv_core
		${GSL_LIBRARIES}
		${LAPACK_LIBRARIES}
		${Boost_TIMER_LIBRARY}
		${Boost_SYSTEM_LIBRARY}
	)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		swl_rnd_util
		swl_math
		swl_base
#		${OpenCV_LIBS}
		opencv_imgproc
		opencv_highgui
		opencv_core
		${GSL_LIBRARIES}
		${LAPACK_LIBRARIES}
		${Boost_TIMER_LIBRARY}
		${Boost_SYSTEM_LIBRARY}
	)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

add_executable(${TARGET} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})

add_test(
	NAME unit_test/${TARGET}
	COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}
	WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)
