#include "swl/Config.h"
#include "swl/rnd_util/HmmWithMultivariateNormalObservations.h"
#include <stdexcept>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

HmmWithMultivariateNormalObservations::HmmWithMultivariateNormalObservations(const size_t K, const size_t D)
: base_type(K, D), mus_(boost::extents[K][D]), sigmas_(boost::extents[K][D][D])  // 0-based index
//: base_type(K, D), mus_(boost::extents[boost::multi_array_types::extent_range(1, K+1)][boost::multi_array_types::extent_range(1, D+1)]), sigmas_(boost::extents[boost::multi_array_types::extent_range(1, K+1)][boost::multi_array_types::extent_range(1, D+1)][boost::multi_array_types::extent_range(1, D+1)])  // 1-based index
{
}

HmmWithMultivariateNormalObservations::HmmWithMultivariateNormalObservations(const size_t K, const size_t D, const std::vector<double> &pi, const boost::multi_array<double, 2> &A, const boost::multi_array<double, 2> &mus, const boost::multi_array<double, 3> &sigmas)
: base_type(K, D, pi, A), mus_(mus), sigmas_(sigmas)
{
}

HmmWithMultivariateNormalObservations::~HmmWithMultivariateNormalObservations()
{
}

void HmmWithMultivariateNormalObservations::doEstimateObservationDensityParametersInMStep(const size_t N, const boost::multi_array<double, 2> &observations, boost::multi_array<double, 2> &gamma, const double denominatorA, const size_t k)
{
	size_t d, n;

	// reestimate symbol prob in each state
	const double denominatorPr = denominatorA + gamma[N-1][k];

	// for multivariate normal distributions
	// TODO [check] >> this code may be changed into a vector form.
	double numeratorPr;
	for (d = 0; d < D_; ++d)
	{
		numeratorPr = 0.0;
		for (n = 0; n < N; ++n)
			numeratorPr += gamma[n][k] * observations[n][d];
		mus_[k][d] = 0.001 + 0.999 * numeratorPr / denominatorPr;
	}

	// for multivariate normal distributions
	// FIXME [modify] >> this code may be changed into a matrix form.
	throw std::runtime_error("this code may be changed into a matrix form.");
/*
	boost::multi_array<double, 3>::array_view<2>::type sigma = sigmas_[boost::indices[k][boost::multi_array<double, 3>::index_range()][boost::multi_array<double, 3>::index_range()]];
	for (d = 0; d < D_; ++d)
	{
		numeratorPr = 0.0;
		for (n = 0; n < N; ++n)
			numeratorPr += gamma[n][k] * (observations[n][d] - mus_[k][d]) * (observations[n][d] - mus_[k][d]).tranpose();
		sigma = 0.001 + 0.999 * numeratorPr / denominatorPr;
	}
*/
}

void HmmWithMultivariateNormalObservations::doEstimateObservationDensityParametersInMStep(const std::vector<size_t> &Ns, const std::vector<boost::multi_array<double, 2> > &observationSequences, const std::vector<boost::multi_array<double, 2> > &gammas, const size_t R, const double denominatorA, const size_t k)
{
	size_t d, n, r;

	// reestimate symbol prob in each state
	double denominatorPr = denominatorA;
	for (r = 0; r < R; ++r)
		denominatorPr += gammas[r][Ns[r]-1][k];

	// for multivariate normal distributions
	// TODO [check] >> this code may be changed into a vector form.
	double numeratorPr;
	for (d = 0; d < D_; ++d)
	{
		numeratorPr = 0.0;
		for (r = 0; r < R; ++r)
			for (n = 0; n < Ns[r]; ++n)
				numeratorPr += gammas[r][n][k] * observationSequences[r][n][d];
		mus_[k][d] = 0.001 + 0.999 * numeratorPr / denominatorPr;
	}

	// for multivariate normal distributions
	// FIXME [modify] >> this code may be changed into a matrix form.
	throw std::runtime_error("this code may be changed into a matrix form.");
/*
	boost::multi_array<double, 3>::array_view<2>::type sigma = sigmas_[boost::indices[k][boost::multi_array<double, 3>::index_range()][boost::multi_array<double, 3>::index_range()]];
	for (d = 0; d < D_; ++d)
	{
		numeratorPr = 0.0;
		for (r = 0; r < R; ++r)
			for (n = 0; n < N; ++n)
				numeratorPr += gammas[r][n][k] * (observationSequences[r][n][d] - mus_[k][d]) * (observationSequences[r][n][d] - mus_[k][d]).tranpose();
		sigma = 0.001 + 0.999 * numeratorPr / denominatorPr;
	}
*/
}

double HmmWithMultivariateNormalObservations::doEvaluateEmissionProbability(const unsigned int state, const boost::multi_array<double, 2>::const_array_view<1>::type &observation) const
{
	throw std::runtime_error("not yet implemented");
}

void HmmWithMultivariateNormalObservations::doGenerateObservationsSymbol(const unsigned int state, boost::multi_array<double, 2>::array_view<1>::type &observation, const unsigned int seed /*= (unsigned int)-1*/) const
{
	throw std::runtime_error("not yet implemented");
}

bool HmmWithMultivariateNormalObservations::doReadObservationDensity(std::istream &stream)
{
	std::runtime_error("not yet implemented");
	return false;
}

bool HmmWithMultivariateNormalObservations::doWriteObservationDensity(std::ostream &stream) const
{
	std::runtime_error("not yet implemented");
	return false;
}

void HmmWithMultivariateNormalObservations::doInitializeObservationDensity()
{
	std::runtime_error("not yet implemented");
}

}  // namespace swl
