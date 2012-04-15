#include "swl/Config.h"
#include "swl/rnd_util/HmmWithVonMisesFisherMixtureObservations.h"
#include <stdexcept>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

HmmWithVonMisesFisherMixtureObservations::HmmWithVonMisesFisherMixtureObservations(const size_t K, const size_t D, const size_t C)
: base_type(K, D), HmmWithMixtureObservations(C), mus_(boost::extents[K][C][D]), kappas_(boost::extents[K][C])  // 0-based index
//: base_type(K, D), HmmWithMixtureObservations(C), mus_(boost::extents[boost::multi_array_types::extent_range(1, K+1)][boost::multi_array_types::extent_range(1, C+1)][boost::multi_array_types::extent_range(1, D+1)]), kappas_(boost::extents[boost::multi_array_types::extent_range(1, K+1)][boost::multi_array_types::extent_range(1, C+1)])  // 1-based index
{
}

HmmWithVonMisesFisherMixtureObservations::HmmWithVonMisesFisherMixtureObservations(const size_t K, const size_t D, const size_t C, const std::vector<double> &pi, const boost::multi_array<double, 2> &A, const std::vector<double> &alpha, const boost::multi_array<double, 3> &mus, const boost::multi_array<double, 2> &kappas)
: base_type(K, D, pi, A), HmmWithMixtureObservations(C, alpha), mus_(mus), kappas_(kappas)
{
}

HmmWithVonMisesFisherMixtureObservations::~HmmWithVonMisesFisherMixtureObservations()
{
}

bool HmmWithVonMisesFisherMixtureObservations::estimateParameters(const size_t N, const boost::multi_array<double, 2> &observations, const double terminationTolerance, boost::multi_array<double, 2> &alpha, boost::multi_array<double, 2> &beta, boost::multi_array<double, 2> &gamma, size_t &numIteration, double &initLogProbability, double &finalLogProbability)
{
	throw std::runtime_error("not yet implemented");
}

double HmmWithVonMisesFisherMixtureObservations::evaluateEmissionProbability(const int state, const boost::multi_array<double, 2>::const_array_view<1>::type &observation) const
{
	throw std::runtime_error("not yet implemented");
}

void HmmWithVonMisesFisherMixtureObservations::generateObservationsSymbol(const int state, boost::multi_array<double, 2>::array_view<1>::type &observation, const bool setSeed /*= false*/) const
{
	throw std::runtime_error("not yet implemented");
}

bool HmmWithVonMisesFisherMixtureObservations::readObservationDensity(std::istream &stream)
{
	std::runtime_error("not yet implemented");
	return false;
}

bool HmmWithVonMisesFisherMixtureObservations::writeObservationDensity(std::ostream &stream) const
{
	std::runtime_error("not yet implemented");
	return false;
}

void HmmWithVonMisesFisherMixtureObservations::initializeObservationDensity()
{
	std::runtime_error("not yet implemented");
}

}  // namespace swl
