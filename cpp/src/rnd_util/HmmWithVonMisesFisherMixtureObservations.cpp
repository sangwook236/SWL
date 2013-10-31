#include "swl/Config.h"
#include "swl/rnd_util/HmmWithVonMisesFisherMixtureObservations.h"
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <stdexcept>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

HmmWithVonMisesFisherMixtureObservations::HmmWithVonMisesFisherMixtureObservations(const size_t K, const size_t D, const size_t C)
: base_type(K, D), HmmWithMixtureObservations(C, K), mus_(boost::extents[K][C]), kappas_(K, C, 0.0),  // 0-based index
  ms_conj_(), Rs_conj_(), cs_conj_()
{
	for (size_t k = 0; k < K; ++k)
		for (size_t c = 0; c < C; ++c)
			mus_[k][c].resize(D);
}

HmmWithVonMisesFisherMixtureObservations::HmmWithVonMisesFisherMixtureObservations(const size_t K, const size_t D, const size_t C, const dvector_type &pi, const dmatrix_type &A, const dmatrix_type &alphas, const boost::multi_array<dvector_type, 2> &mus, const dmatrix_type &kappas)
: base_type(K, D, pi, A), HmmWithMixtureObservations(C, K, alphas), mus_(mus), kappas_(kappas),
  ms_conj_(), Rs_conj_(), cs_conj_()
{
}

HmmWithVonMisesFisherMixtureObservations::HmmWithVonMisesFisherMixtureObservations(const size_t K, const size_t D, const size_t C, const dvector_type *pi_conj, const dmatrix_type *A_conj, const dmatrix_type *alphas_conj, const boost::multi_array<dvector_type, 2> *ms_conj, const dmatrix_type *Rs_conj, const dmatrix_type *cs_conj)
: base_type(K, D, pi_conj, A_conj), HmmWithMixtureObservations(C, K, alphas_conj), mus_(boost::extents[K][C]), kappas_(K, C, 0.0),
  ms_conj_(ms_conj), Rs_conj_(Rs_conj), cs_conj_(cs_conj)
{
}

HmmWithVonMisesFisherMixtureObservations::~HmmWithVonMisesFisherMixtureObservations()
{
}

void HmmWithVonMisesFisherMixtureObservations::doEstimateObservationDensityParametersByML(const size_t N, const unsigned int state, const dmatrix_type &observations, const dmatrix_type &gamma, const double denominatorA)
{
	throw std::runtime_error("not yet implemented");
}

void HmmWithVonMisesFisherMixtureObservations::doEstimateObservationDensityParametersByML(const std::vector<size_t> &Ns, const unsigned int state, const std::vector<dmatrix_type> &observationSequences, const std::vector<dmatrix_type> &gammas, const size_t R, const double denominatorA)
{
	throw std::runtime_error("not yet implemented");
}

void HmmWithVonMisesFisherMixtureObservations::doEstimateObservationDensityParametersByMAPUsingConjugatePrior(const size_t N, const unsigned int state, const dmatrix_type &observations, const dmatrix_type &gamma, const double denominatorA)
{
	throw std::runtime_error("not yet implemented");
}

void HmmWithVonMisesFisherMixtureObservations::doEstimateObservationDensityParametersByMAPUsingConjugatePrior(const std::vector<size_t> &Ns, const unsigned int state, const std::vector<dmatrix_type> &observationSequences, const std::vector<dmatrix_type> &gammas, const size_t R, const double denominatorA)
{
	throw std::runtime_error("not yet implemented");
}

void HmmWithVonMisesFisherMixtureObservations::doEstimateObservationDensityParametersByMAPUsingEntropicPrior(const size_t N, const unsigned int state, const dmatrix_type &observations, const dmatrix_type &gamma, const double z, const bool doesTrimParameter, const double terminationTolerance, const size_t maxIteration, const double /*denominatorA*/)
{
	throw std::runtime_error("not yet implemented");
}

void HmmWithVonMisesFisherMixtureObservations::doEstimateObservationDensityParametersByMAPUsingEntropicPrior(const std::vector<size_t> &Ns, const unsigned int state, const std::vector<dmatrix_type> &observationSequences, const std::vector<dmatrix_type> &gammas, const double z, const bool doesTrimParameter, const double terminationTolerance, const size_t maxIteration, const size_t R, const double /*denominatorA*/)
{
	throw std::runtime_error("not yet implemented");
}

double HmmWithVonMisesFisherMixtureObservations::doEvaluateEmissionProbability(const unsigned int state, const dvector_type &observation) const
{
	throw std::runtime_error("not yet implemented");
}

void HmmWithVonMisesFisherMixtureObservations::doGenerateObservationsSymbol(const unsigned int state, boost::numeric::ublas::matrix_row<dmatrix_type> &observation) const
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

void HmmWithVonMisesFisherMixtureObservations::doInitializeObservationDensity(const std::vector<double> &lowerBoundsOfObservationDensity, const std::vector<double> &upperBoundsOfObservationDensity)
{
	std::runtime_error("not yet implemented");
}

}  // namespace swl
