#include "swl/Config.h"
#include "swl/rnd_util/HmmWithVonMisesFisherObservations.h"
#include <stdexcept>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

HmmWithVonMisesFisherObservations::HmmWithVonMisesFisherObservations(const size_t K, const size_t D)
: base_type(K, D), mus_(boost::extents[K][D]), kappas_(K, 0.0)  // 0-based index
//: base_type(K, D), mus_(boost::extents[boost::multi_array_types::extent_range(1, K+1)][boost::multi_array_types::extent_range(1, D+1)]), kappas_(K, 0.0)  // 1-based index (???)
{
}

HmmWithVonMisesFisherObservations::HmmWithVonMisesFisherObservations(const size_t K, const size_t D, const std::vector<double> &pi, const boost::multi_array<double, 2> &A, const boost::multi_array<double, 2> &mus, const std::vector<double> &kappas)
: base_type(K, D, pi, A), mus_(mus), kappas_(kappas)
{
}

HmmWithVonMisesFisherObservations::~HmmWithVonMisesFisherObservations()
{
}

void HmmWithVonMisesFisherObservations::doEstimateObservationDensityParametersInMStep(const size_t N, const unsigned int state, const boost::multi_array<double, 2> &observations, boost::multi_array<double, 2> &gamma, const double denominatorA)
{
	throw std::runtime_error("not yet implemented");
}

void HmmWithVonMisesFisherObservations::doEstimateObservationDensityParametersInMStep(const std::vector<size_t> &Ns, const unsigned int state, const std::vector<boost::multi_array<double, 2> > &observationSequences, const std::vector<boost::multi_array<double, 2> > &gammas, const size_t R, const double denominatorA)
{
	throw std::runtime_error("not yet implemented");
}

double HmmWithVonMisesFisherObservations::doEvaluateEmissionProbability(const unsigned int state, const boost::multi_array<double, 2>::const_array_view<1>::type &observation) const
{
	throw std::runtime_error("not yet implemented");
}

void HmmWithVonMisesFisherObservations::doGenerateObservationsSymbol(const unsigned int state, boost::multi_array<double, 2>::array_view<1>::type &observation, const unsigned int seed /*= (unsigned int)-1*/) const
{
	throw std::runtime_error("not yet implemented");
}

bool HmmWithVonMisesFisherObservations::doReadObservationDensity(std::istream &stream)
{
	std::runtime_error("not yet implemented");
	return false;
}

bool HmmWithVonMisesFisherObservations::doWriteObservationDensity(std::ostream &stream) const
{
	std::runtime_error("not yet implemented");
	return false;
}

void HmmWithVonMisesFisherObservations::doInitializeObservationDensity()
{
	std::runtime_error("not yet implemented");
}

}  // namespace swl
