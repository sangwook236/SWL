#include "swl/Config.h"
#include "swl/rnd_util/HmmWithMultivariateGaussianMixtureObservations.h"
#include <stdexcept>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

HmmWithMultivariateGaussianMixtureObservations::HmmWithMultivariateGaussianMixtureObservations(const size_t K, const size_t D, const size_t C)
: base_type(K, D), HmmWithMixtureObservations(C), mus_(boost::extents[K][C][D]), sigmas_(boost::extents[K][C][D][D])  // 0-based index
//: base_type(K, D), HmmWithMixtureObservations(C), mus_(boost::extents[boost::multi_array_types::extent_range(1, K+1)][boost::multi_array_types::extent_range(1, C+1)][boost::multi_array_types::extent_range(1, D+1)]), sigmas_(boost::extents[boost::multi_array_types::extent_range(1, K+1)][boost::multi_array_types::extent_range(1, C+1)][boost::multi_array_types::extent_range(1, D+1)][boost::multi_array_types::extent_range(1, D+1)])  // 1-based index
{
}

HmmWithMultivariateGaussianMixtureObservations::HmmWithMultivariateGaussianMixtureObservations(const size_t K, const size_t D, const size_t C, const std::vector<double> &pi, const boost::multi_array<double, 2> &A, const std::vector<double> &alpha, const boost::multi_array<double, 3> &mus, const boost::multi_array<double, 4> &sigmas)
: base_type(K, D, pi, A), HmmWithMixtureObservations(C, alpha), mus_(mus), sigmas_(sigmas)
{
}

HmmWithMultivariateGaussianMixtureObservations::~HmmWithMultivariateGaussianMixtureObservations()
{
}

bool HmmWithMultivariateGaussianMixtureObservations::estimateParameters(const size_t N, const boost::multi_array<double, 2> &observations, const double terminationTolerance, boost::multi_array<double, 2> &alpha, boost::multi_array<double, 2> &beta, boost::multi_array<double, 2> &gamma, size_t &numIteration, double &initLogProbability, double &finalLogProbability)
{
	throw std::runtime_error("not yet implemented");
}

bool HmmWithMultivariateGaussianMixtureObservations::estimateParameters(const std::vector<size_t> &Ns, const std::vector<boost::multi_array<double, 2> > &observationSequences, const double terminationTolerance, size_t &numIteration,std::vector<double> &initLogProbabilities, std::vector<double> &finalLogProbabilities)
{
	throw std::runtime_error("not yet implemented");
}

double HmmWithMultivariateGaussianMixtureObservations::evaluateEmissionProbability(const unsigned int state, const boost::multi_array<double, 2>::const_array_view<1>::type &observation) const
{
	throw std::runtime_error("not yet implemented");
}

void HmmWithMultivariateGaussianMixtureObservations::generateObservationsSymbol(const unsigned int state, boost::multi_array<double, 2>::array_view<1>::type &observation, const unsigned int seed /*= (unsigned int)-1*/) const
{
	throw std::runtime_error("not yet implemented");
}

bool HmmWithMultivariateGaussianMixtureObservations::readObservationDensity(std::istream &stream)
{
	std::runtime_error("not yet implemented");
	return false;
}

bool HmmWithMultivariateGaussianMixtureObservations::writeObservationDensity(std::ostream &stream) const
{
	std::runtime_error("not yet implemented");
	return false;
}

void HmmWithMultivariateGaussianMixtureObservations::initializeObservationDensity()
{
	std::runtime_error("not yet implemented");
}

}  // namespace swl
