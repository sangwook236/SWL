if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_rnd_util_d)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(TARGET swl_rnd_util)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

set(SRCS
	ArHmmWithMultivariateNormalMixtureObservations.cpp
	ArHmmWithMultivariateNormalObservations.cpp
	ArHmmWithUnivariateNormalMixtureObservations.cpp
	ArHmmWithUnivariateNormalObservations.cpp
	AutoRegression.cpp
	CDHMM.cpp
	CDHMMWithMixtureObservations.cpp
	ContinuousDensityMixtureModel.cpp
	DDHMM.cpp
	ExtendedKalmanFilter.cpp
	HistogramAccumulator.cpp
	HistogramMatcher.cpp
	HistogramUitl.cpp
	HMM.cpp
	HmmSegmenter.cpp
	HmmWithMultinomialObservations.cpp
	HmmWithMultivariateNormalMixtureObservations.cpp
	HmmWithMultivariateNormalObservations.cpp
	HmmWithUnivariateNormalMixtureObservations.cpp
	HmmWithUnivariateNormalObservations.cpp
	HmmWithVonMisesFisherMixtureObservations.cpp
	HmmWithVonMisesFisherObservations.cpp
	HmmWithVonMisesMixtureObservations.cpp
	HmmWithVonMisesObservations.cpp
	HoughTransform.cpp
	KalmanFilter.cpp
	LambertWFunction.cpp
	LevenshteinDistance.cpp
	MetropolisHastingsAlgorithm.cpp
	MixtureModel.cpp
	MultivariateNormalMixtureModel.cpp
	Ransac.cpp
	RejectionSampling.cpp
	RndUtilLocalApi.cpp
	SamplingImportanceResampling.cpp
	UnivariateNormalMixtureModel.cpp
	UnscentedKalmanFilter.cpp
	UnscentedKalmanFilterWithAdditiveNoise.cpp
	VonMisesMixtureModel.cpp
)
if(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		swl_math_d
		swl_base_d
		${OpenCV_LIBS}
		${GSL_LIBRARIES}
		${LAPACK_LIBRARIES}
	)
else(CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(LIBS
		swl_math
		swl_base
		${OpenCV_LIBS}
		${GSL_LIBRARIES}
		${LAPACK_LIBRARIES}
	)
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

add_definitions(-DSWL_RND_UTIL_EXPORT)

add_library(${TARGET} ${LIB_TYPE} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})
