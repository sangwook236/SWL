#include "swl/Config.h"
#include "swl/rnd_util/HmmWithVonMisesFisherMixtureObservations.h"
#include <stdexcept>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

HmmWithVonMisesFisherMixtureObservations::HmmWithVonMisesFisherMixtureObservations(const size_t K, const size_t D, const size_t C)
: base_type(K, D), HmmWithMixtureObservations(C, K), mus_(boost::extents[K][C][D]), kappas_(boost::extents[K][C])  // 0-based index
//: base_type(K, D), HmmWithMixtureObservations(C, K), mus_(boost::extents[boost::multi_array_types::extent_range(1, K+1)][boost::multi_array_types::extent_range(1, C+1)][boost::multi_array_types::extent_range(1, D+1)]), kappas_(boost::extents[boost::multi_array_types::extent_range(1, K+1)][boost::multi_array_types::extent_range(1, C+1)])  // 1-based index
{
}

HmmWithVonMisesFisherMixtureObservations::HmmWithVonMisesFisherMixtureObservations(const size_t K, const size_t D, const size_t C, const std::vector<double> &pi, const boost::multi_array<double, 2> &A, const boost::multi_array<double, 2> &alphas, const boost::multi_array<double, 3> &mus, const boost::multi_array<double, 2> &kappas)
: base_type(K, D, pi, A), HmmWithMixtureObservations(C, K, alphas), mus_(mus), kappas_(kappas)
{
}

HmmWithVonMisesFisherMixtureObservations::~HmmWithVonMisesFisherMixtureObservations()
{
}

void HmmWithVonMisesFisherMixtureObservations::doEstimateObservationDensityParametersInMStep(const size_t N, const unsigned int state, const boost::multi_array<double, 2> &observations, boost::multi_array<double, 2> &gamma, const double denominatorA)
{
	throw std::runtime_error("not yet implemented");
}

void HmmWithVonMisesFisherMixtureObservations::doEstimateObservationDensityParametersInMStep(const std::vector<size_t> &Ns, const unsigned int state, const std::vector<boost::multi_array<double, 2> > &observationSequences, const std::vector<boost::multi_array<double, 2> > &gammas, const size_t R, const double denominatorA)
{
	throw std::runtime_error("not yet implemented");
}

double HmmWithVonMisesFisherMixtureObservations::doEvaluateEmissionProbability(const unsigned int state, const boost::multi_array<double, 2>::const_array_view<1>::type &observation) const
{
	throw std::runtime_error("not yet implemented");
}

void HmmWithVonMisesFisherMixtureObservations::doGenerateObservationsSymbol(const unsigned int state, boost::multi_array<double, 2>::array_view<1>::type &observation, const unsigned int seed /*= (unsigned int)-1*/) const
{
	throw std::runtime_error("not yet implemented");
}

bool HmmWithVonMisesFisherMixtureObservations::doReadObservationDensity(std::istream &stream)
{
	std::runtime_error("not yet implemented");
	return false;
}

bool HmmWithVonMisesFisherMixtureObservations::doWriteObservationDensity(std::ostream &stream) const
{
	std::runtime_error("not yet implemented");
	return false;
}

void HmmWithVonMisesFisherMixtureObservations::doInitializeObservationDensity()
{
	std::runtime_error("not yet implemented");
}

}  // namespace swl
