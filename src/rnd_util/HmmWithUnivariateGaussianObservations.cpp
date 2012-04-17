#include "swl/Config.h"
#include "swl/rnd_util/HmmWithUnivariateGaussianObservations.h"
#include <boost/math/distributions/normal.hpp>  // for normal distribution
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <ctime>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

HmmWithUnivariateGaussianObservations::HmmWithUnivariateGaussianObservations(const size_t K)
: base_type(K, 1), mus_(K, 0.0), sigmas_(K, 0.0),  // 0-based index
  baseGenerator_()
{
}

HmmWithUnivariateGaussianObservations::HmmWithUnivariateGaussianObservations(const size_t K, const std::vector<double> &pi, const boost::multi_array<double, 2> &A, const std::vector<double> &mus, const std::vector<double> &sigmas)
: base_type(K, 1, pi, A), mus_(mus), sigmas_(sigmas),
  baseGenerator_()
{
}

HmmWithUnivariateGaussianObservations::~HmmWithUnivariateGaussianObservations()
{
}

bool HmmWithUnivariateGaussianObservations::estimateParameters(const size_t N, const boost::multi_array<double, 2> &observations, const double terminationTolerance, boost::multi_array<double, 2> &alpha, boost::multi_array<double, 2> &beta, boost::multi_array<double, 2> &gamma, size_t &numIteration, double &initLogProbability, double &finalLogProbability)
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
			// reestimate frequency of state k in time n=0
			pi_[k] = .001 + .999 * gamma[0][k];

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
			denominatorP = denominatorA + gamma[N-1][k];

			// for univariate normal distributions
			numeratorP = 0.0;
			for (n = 0; n < N; ++n)
				numeratorP += gamma[n][k] * observations[n][0];
			mus_[k] = .001 + .999 * numeratorP / denominatorP;

			// for univariate normal distributions
			numeratorP = 0.0;
			for (n = 0; n < N; ++n)
				numeratorP += gamma[n][k] * (observations[n][0] - mus_[k]) * (observations[n][0] - mus_[k]);
			sigmas_[k] = .001 + .999 * numeratorP / denominatorP;
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

double HmmWithUnivariateGaussianObservations::evaluateEmissionProbability(const unsigned int state, const boost::multi_array<double, 2>::const_array_view<1>::type &observation) const
{
	//boost::math::normal pdf;  // (default mean = zero, and standard deviation = unity)
	boost::math::normal pdf(mus_[state], sigmas_[state]);

	return boost::math::pdf(pdf, observation[0]);
}

void HmmWithUnivariateGaussianObservations::generateObservationsSymbol(const unsigned int state, boost::multi_array<double, 2>::array_view<1>::type &observation, const unsigned int seed /*= (unsigned int)-1*/) const
{
	typedef boost::normal_distribution<> distribution_type;
	typedef boost::variate_generator<base_generator_type &, distribution_type> generator_type;

	if ((unsigned int)-1 != seed)
		baseGenerator_.seed(seed);

	generator_type normal_gen(baseGenerator_, distribution_type(mus_[state], sigmas_[state]));
	for (size_t i = 0; i < D_; ++i)
		observation[i] = normal_gen();
}

bool HmmWithUnivariateGaussianObservations::readObservationDensity(std::istream &stream)
{
	if (1 != D_) return false;

	std::string dummy;
	stream >> dummy;
#if defined(__GNUC__)
	if (strcasecmp(dummy.c_str(), "univariate") != 0)
#elif defined(_MSC_VER)
	if (_stricmp(dummy.c_str(), "univariate") != 0)
#endif
		return false;

	stream >> dummy;
#if defined(__GNUC__)
	if (strcasecmp(dummy.c_str(), "normal:") != 0)
#elif defined(_MSC_VER)
	if (_stricmp(dummy.c_str(), "normal:") != 0)
#endif
		return false;

	for (size_t k = 0; k < K_; ++k)
		stream >> mus_[k] >> sigmas_[k];

	return true;
}

bool HmmWithUnivariateGaussianObservations::writeObservationDensity(std::ostream &stream) const
{
	stream << "univariate normal:" << std::endl;
	for (size_t k = 0; k < K_; ++k)
		stream << mus_[k] << ' ' << sigmas_[k] << std::endl;

	return true;
}

void HmmWithUnivariateGaussianObservations::initializeObservationDensity()
{
	// PRECONDITIONS [] >>
	//	-. std::srand() had to be called before this function is called.

	const double lb = -10000.0, ub = 10000.0;
	for (size_t k = 0; k < K_; ++k)
	{
		mus_[k] = ((double)std::rand() / RAND_MAX) * (ub - lb) + lb;
		sigmas_[k] = ((double)std::rand() / RAND_MAX) * (ub - lb) + lb;
	}
}

}  // namespace swl
