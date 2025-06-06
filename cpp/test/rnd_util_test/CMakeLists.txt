set(TARGET swl_rnd_util_test)

set(SRCS
	main.cpp
	ArHmmWithUnivariateNormalMixtureObservationsTest.cpp
	ArHmmWithUnivariateNormalObservationsTest.cpp
	DynamicTimeWarpingTest.cpp
	ExtendedKalmanFilterTest.cpp
	GraphAlgorithmTest.cpp
	GraphTraversalTest.cpp
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
	SignalProcessingTest.cpp
	UnivariateNormalMixtureModelTest.cpp
	UnscentedKalmanFilterTest.cpp
	UnscentedKalmanFilterWithAdditiveNoiseTest.cpp
	VonMisesMixtureModelTest.cpp
)
set(LIBS
	swl_rnd_util
	swl_math
	swl_base
	${OpenCV_LIBRARIES}
	${GSL_LIBRARIES}
	${LAPACK_LIBRARIES}
	${Boost_TIMER_LIBRARY}
	${Boost_SYSTEM_LIBRARY}
)

add_executable(${TARGET} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})

if(NOT USE_UNIT_TESTCASES_ONLY_AS_TESTS)
	add_test(
		NAME test/${TARGET}
		COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}
		WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
	)
endif(NOT USE_UNIT_TESTCASES_ONLY_AS_TESTS)
