set(TARGET swl_rnd_util)

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
set(LIBS
	swl_math
	swl_base
	${OpenCV2_LIBRARIES}
	${GSL_LIBRARIES}
	${LAPACK_LIBRARIES}
)

add_definitions(-DSWL_RND_UTIL_EXPORT)

if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
	add_compile_options(-fPIC)
endif(CMAKE_CXX_COMPILER_ID MATCHES GNU)

add_library(${TARGET} ${LIB_TYPE} ${SRCS})
target_link_libraries(${TARGET} ${LIBS})
