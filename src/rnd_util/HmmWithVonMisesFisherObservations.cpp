#include "swl/Config.h"
#include "swl/rnd_util/HmmWithVonMisesFisherObservations.h"
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <stdexcept>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

HmmWithVonMisesFisherObservations::HmmWithVonMisesFisherObservations(const size_t K, const size_t D)
: base_type(K, D), mus_(K, D), kappas_(K, 0.0)  // 0-based index
{
}

HmmWithVonMisesFisherObservations::HmmWithVonMisesFisherObservations(const size_t K, const size_t D, const dvector_type &pi, const dmatrix_type &A, const dmatrix_type &mus, const dvector_type &kappas)
: base_type(K, D, pi, A), mus_(mus), kappas_(kappas)
{
}

HmmWithVonMisesFisherObservations::~HmmWithVonMisesFisherObservations()
{
}

void HmmWithVonMisesFisherObservations::doEstimateObservationDensityParametersByML(const size_t N, const unsigned int state, const dmatrix_type &observations, dmatrix_type &gamma, const double denominatorA)
{
	throw std::runtime_error("not yet implemented");
}

void HmmWithVonMisesFisherObservations::doEstimateObservationDensityParametersByML(const std::vector<size_t> &Ns, const unsigned int state, const std::vector<dmatrix_type> &observationSequences, const std::vector<dmatrix_type> &gammas, const size_t R, const double denominatorA)
{
	throw std::runtime_error("not yet implemented");
}

double HmmWithVonMisesFisherObservations::doEvaluateEmissionProbability(const unsigned int state, const boost::numeric::ublas::matrix_row<const dmatrix_type> &observation) const
{
	throw std::runtime_error("not yet implemented");
}

void HmmWithVonMisesFisherObservations::doGenerateObservationsSymbol(const unsigned int state, boost::numeric::ublas::matrix_row<dmatrix_type> &observation, const unsigned int seed /*= (unsigned int)-1*/) const
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

void HmmWithVonMisesFisherObservations::doInitializeObservationDensity(const std::vector<double> &lowerBoundsOfObservationDensity, const std::vector<double> &upperBoundsOfObservationDensity)
{
	std::runtime_error("not yet implemented");
}

}  // namespace swl
