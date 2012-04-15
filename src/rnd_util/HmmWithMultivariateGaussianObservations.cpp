#include "swl/Config.h"
#include "swl/rnd_util/HmmWithMultivariateGaussianObservations.h"
#include <stdexcept>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

HmmWithMultivariateGaussianObservations::HmmWithMultivariateGaussianObservations(const size_t K, const size_t D)
: base_type(K, D), mus_(boost::extents[K][D]), sigmas_(boost::extents[K][D][D])  // 0-based index
//: base_type(K, D), mus_(boost::extents[boost::multi_array_types::extent_range(1, K+1)][boost::multi_array_types::extent_range(1, D+1)]), sigmas_(boost::extents[boost::multi_array_types::extent_range(1, K+1)][boost::multi_array_types::extent_range(1, D+1)][boost::multi_array_types::extent_range(1, D+1)])  // 1-based index
{
}

HmmWithMultivariateGaussianObservations::HmmWithMultivariateGaussianObservations(const size_t K, const size_t D, const std::vector<double> &pi, const boost::multi_array<double, 2> &A, const boost::multi_array<double, 2> &mus, const boost::multi_array<double, 3> &sigmas)
: base_type(K, D, pi, A), mus_(mus), sigmas_(sigmas)
{
}

HmmWithMultivariateGaussianObservations::~HmmWithMultivariateGaussianObservations()
{
}

bool HmmWithMultivariateGaussianObservations::estimateParameters(const size_t N, const boost::multi_array<double, 2> &observations, const double terminationTolerance, boost::multi_array<double, 2> &alpha, boost::multi_array<double, 2> &beta, boost::multi_array<double, 2> &gamma, size_t &numIteration, double &initLogProbability, double &finalLogProbability)
{
	std::vector<double> scale(N, 0.0);

	double logprobf, logprobb;
	runForwardAlgorithm(N, observations, scale, alpha, logprobf);
	runBackwardAlgorithm(N, observations, scale, beta, logprobb);

	computeGamma(N, alpha, beta, gamma);
	boost::multi_array<double, 3> xi(boost::extents[N][K_][K_]);
	computeXi(N, observations, alpha, beta, xi);

	initLogProbability = logprobf;  // log P(O | initial model)

	double numeratorA, denominatorA;
	double numeratorP, denominatorP;
	double delta, logprobprev = logprobf;
	size_t i, k, n;
	size_t iter = 0;
	do
	{
		for (k = 0; k < K_; ++k)
		{
			// reestimate frequency of state k in time n=1
			pi_[k] = .001 + .999 * gamma[1][k];

			// reestimate transition matrix 
			denominatorA = 0.0;
			for (n = 0; n < N - 1; ++n)
				denominatorA += gamma[n][k];

			for (i = 0; i < K_; ++i)
			{
				numeratorA = 0.0;
				for (n = 0; n < N - 1; ++n)
					numeratorA += xi[n][k][i];
				A_[k][i] = .001 + .999 * numeratorA / denominatorA;
			}

			// reestimate symbol prob in each state
			denominatorP = denominatorA + gamma[N][k];

			// for multivariate normal distributions
			// TODO [check] >> this code may be changed into a vector form.
			for (i = 0; i < D_; ++i)
			{
				numeratorP = 0.0;
				for (n = 0; n < N; ++n)
					numeratorP += gamma[n][k] * observations[n][i];
				mus_[k][i] = .001 + .999 * numeratorP / denominatorP;
			}

			// for multivariate normal distributions
			// FIXME [modify] >> this code may be changed into a matrix form.
			throw std::runtime_error("this code may be changed into a matrix form.");
/*
			boost::multi_array<double, 3>::array_view<2>::type sigma = sigmas_[boost::indices[k][boost::multi_array<double, 3>::index_range()][boost::multi_array<double, 3>::index_range()]];
			for (i = 0; i < D_; ++i)
			{
				numeratorP = 0.0;
				for (n = 0; n < N; ++n)
					numeratorP += gamma[n][k] * (observations[n][i] - mus_[k][i]) * (observations[n][i] - mus_[k][i]).tranpose();
				sigma = .001 + .999 * numeratorP / denominatorP;
			}
*/
		}

		runForwardAlgorithm(N, observations, scale, alpha, logprobf);
		runBackwardAlgorithm(N, observations, scale, beta, logprobb);

		computeGamma(N, alpha, beta, gamma);
		computeXi(N, observations, alpha, beta, xi);

		// compute difference between log probability of two iterations
		delta = logprobf - logprobprev;
		logprobprev = logprobf;
		++iter;
	} while (delta > terminationTolerance);  // if log probability does not change much, exit

	numIteration = iter;
	finalLogProbability = logprobf;  // log P(observations | estimated model)

	return true;
}

double HmmWithMultivariateGaussianObservations::evaluateEmissionProbability(const int state, const boost::multi_array<double, 2>::const_array_view<1>::type &observation) const
{
	throw std::runtime_error("not yet implemented");
}

void HmmWithMultivariateGaussianObservations::generateObservationsSymbol(const int state, boost::multi_array<double, 2>::array_view<1>::type &observation, const bool setSeed /*= false*/) const
{
	throw std::runtime_error("not yet implemented");
}

bool HmmWithMultivariateGaussianObservations::readObservationDensity(std::istream &stream)
{
	std::runtime_error("not yet implemented");
	return false;
}

bool HmmWithMultivariateGaussianObservations::writeObservationDensity(std::ostream &stream) const
{
	std::runtime_error("not yet implemented");
	return false;
}

void HmmWithMultivariateGaussianObservations::initializeObservationDensity()
{
	std::runtime_error("not yet implemented");
}

}  // namespace swl
