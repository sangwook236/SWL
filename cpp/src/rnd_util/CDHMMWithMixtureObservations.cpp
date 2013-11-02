#include "swl/Config.h"
#include "swl/rnd_util/CDHMMWithMixtureObservations.h"
#include <boost/numeric/ublas/matrix_proxy.hpp>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

CDHMMWithMixtureObservations::CDHMMWithMixtureObservations(const size_t K, const size_t D, const size_t C)
: base_type(K, D), C_(C), alphas_(K, C, 0.0),  // 0-based index.
  alphas_conj_()
{
}

CDHMMWithMixtureObservations::CDHMMWithMixtureObservations(const size_t K, const size_t D, const size_t C, const dvector_type &pi, const dmatrix_type &A, const dmatrix_type &alphas)
: base_type(K, D, pi, A), C_(C), alphas_(alphas),
  alphas_conj_()
{
}

CDHMMWithMixtureObservations::CDHMMWithMixtureObservations(const size_t K, const size_t D, const size_t C, const dvector_type *pi_conj, const dmatrix_type *A_conj, const dmatrix_type *alphas_conj)
: base_type(K, D, pi_conj, A_conj), C_(C), alphas_(K, C, 0.0),
  alphas_conj_(alphas_conj)
{
}

CDHMMWithMixtureObservations::~CDHMMWithMixtureObservations()
{
}

double CDHMMWithMixtureObservations::doEvaluateEmissionProbability(const unsigned int state, const dvector_type &observation) const
{
	const double eps = 1e-50;
	double prob = 0.0;
	for (size_t c = 0; c < C_; ++c)
		// TODO [check] >> we need to check if a component is trimmed or not.
		//	Here, we use the value of alpha in order to check if a component is trimmed or not.
		if (std::fabs(alphas_(state, c)) >= eps)
			prob += (alphas_(state, c) * doEvaluateEmissionMixtureComponentProbability(state, c, observation));

	return prob;
}

double CDHMMWithMixtureObservations::doEvaluateEmissionProbability(const unsigned int state, const size_t n, const dmatrix_type &observations) const
{
	const double eps = 1e-50;
	double prob = 0.0;
	for (size_t c = 0; c < C_; ++c)
		// TODO [check] >> we need to check if a component is trimmed or not.
		//	Here, we use the value of alpha in order to check if a component is trimmed or not.
		if (std::fabs(alphas_(state, c)) >= eps)
			prob += (alphas_(state, c) * doEvaluateEmissionMixtureComponentProbability(state, c, n, observations));

	return prob;
}

double CDHMMWithMixtureObservations::doEvaluateEmissionMixtureComponentProbability(const unsigned int state, const unsigned int component, const size_t n, const dmatrix_type &observations) const
{
	return doEvaluateEmissionMixtureComponentProbability(state, component, boost::numeric::ublas::matrix_row<const dmatrix_type>(observations, n));
}

void CDHMMWithMixtureObservations::normalizeObservationDensityParameters(const size_t K)
{
	size_t c;
	double sum;

	for (size_t k = 0; k < K; ++k)
	{
		sum = 0.0;
		for (c = 0; c < C_; ++c)
			sum += alphas_(k, c);
		for (c = 0; c < C_; ++c)
			alphas_(k, c) /= sum;
	}
}

}  // namespace swl
