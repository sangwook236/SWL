#include "swl/Config.h"
#include "swl/rnd_util/HmmWithMultivariateNormalMixtureObservations.h"
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <stdexcept>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

HmmWithMultivariateNormalMixtureObservations::HmmWithMultivariateNormalMixtureObservations(const size_t K, const size_t D, const size_t C)
: base_type(K, D), HmmWithMixtureObservations(C, K), mus_(boost::extents[K][C]), sigmas_(boost::extents[K][C])  // 0-based index
{
	for (size_t k = 0; k < K; ++k)
		for (size_t c = 0; c < C; ++c)
		{
			mus_[k][c].resize(D);
			sigmas_[k][c].resize(D, D);
		}
}

HmmWithMultivariateNormalMixtureObservations::HmmWithMultivariateNormalMixtureObservations(const size_t K, const size_t D, const size_t C, const dvector_type &pi, const dmatrix_type &A, const dmatrix_type &alphas, const boost::multi_array<dvector_type, 2> &mus, const boost::multi_array<dmatrix_type, 2> &sigmas)
: base_type(K, D, pi, A), HmmWithMixtureObservations(C, K, alphas), mus_(mus), sigmas_(sigmas)
{
}

HmmWithMultivariateNormalMixtureObservations::~HmmWithMultivariateNormalMixtureObservations()
{
}

void HmmWithMultivariateNormalMixtureObservations::doEstimateObservationDensityParametersInMStep(const size_t N, const unsigned int state, const dmatrix_type &observations, dmatrix_type &gamma, const double denominatorA)
{
	throw std::runtime_error("not yet implemented");
}

void HmmWithMultivariateNormalMixtureObservations::doEstimateObservationDensityParametersInMStep(const std::vector<size_t> &Ns, const unsigned int state, const std::vector<dmatrix_type> &observationSequences, const std::vector<dmatrix_type> &gammas, const size_t R, const double denominatorA)
{
	throw std::runtime_error("not yet implemented");
}

double HmmWithMultivariateNormalMixtureObservations::doEvaluateEmissionProbability(const unsigned int state, const boost::numeric::ublas::matrix_row<const dmatrix_type> &observation) const
{
	throw std::runtime_error("not yet implemented");
}

void HmmWithMultivariateNormalMixtureObservations::doGenerateObservationsSymbol(const unsigned int state, boost::numeric::ublas::matrix_row<dmatrix_type> &observation, const unsigned int seed /*= (unsigned int)-1*/) const
{
	throw std::runtime_error("not yet implemented");
}

bool HmmWithMultivariateNormalMixtureObservations::doReadObservationDensity(std::istream &stream)
{
	std::runtime_error("not yet implemented");
	return false;
}

bool HmmWithMultivariateNormalMixtureObservations::doWriteObservationDensity(std::ostream &stream) const
{
	std::runtime_error("not yet implemented");
	return false;
}

void HmmWithMultivariateNormalMixtureObservations::doInitializeObservationDensity()
{
	std::runtime_error("not yet implemented");
}

}  // namespace swl
