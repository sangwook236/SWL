#include "swl/Config.h"
#include "RndUtilLocalApi.h"
#include "swl/rnd_util/ContinuousDensityMixtureModel.h"
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <numeric>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <cassert>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

ContinuousDensityMixtureModel::ContinuousDensityMixtureModel(const std::size_t K, const std::size_t D)
: base_type(K, D)
{
}

ContinuousDensityMixtureModel::ContinuousDensityMixtureModel(const std::size_t K, const std::size_t D, const std::vector<double> &pi)
: base_type(K, D, pi)
{
}

ContinuousDensityMixtureModel::ContinuousDensityMixtureModel(const std::size_t K, const std::size_t D, const std::vector<double> *pi_conj)
: base_type(K, D, pi_conj)
{
}

ContinuousDensityMixtureModel::~ContinuousDensityMixtureModel()
{
}

void ContinuousDensityMixtureModel::computeGamma(const std::size_t N, const dmatrix_type &observations, dmatrix_type &gamma, double &logProbability) const
{
	std::size_t k;
	double denominator;

	logProbability = 0.0;
	for (std::size_t n = 0; n < N; ++n)
	{
		denominator = 0.0;
		for (k = 0; k < K_; ++k)
		{
			//gamma(n, k) = pi_[k] * evaluateEmissionProbability(k, boost::numeric::ublas::matrix_row<const dmatrix_type>(observations, n));  // error !!!
			gamma(n, k) = pi_[k] * doEvaluateMixtureComponentProbability(k, boost::numeric::ublas::matrix_row<const dmatrix_type>(observations, n));
			denominator += gamma(n, k);
		}

		logProbability += std::log(denominator);

		for (k = 0; k < K_; ++k)
			gamma(n, k) /= denominator;
	}
}

bool ContinuousDensityMixtureModel::trainByML(const std::size_t N, const dmatrix_type &observations, const double terminationTolerance, const std::size_t maxIteration, std::size_t &numIteration, double &initLogProbability, double &finalLogProbability)
{
	dmatrix_type gamma(N, K_, 0.0);
	double logProb;

	// E-step: evaluate gamma.
	computeGamma(N, observations, gamma, logProb);

	initLogProbability = logProb;  // log P(observations | initial model).
	finalLogProbability = logProb;

	double delta;
	std::size_t k, n;
	numIteration = 0;
	double sumGamma;
	do
	{
		// M-step.
		for (k = 0; k < K_; ++k)
		{
			// reestimate the mixture coefficient of state k.
			sumGamma = 0.0;
			for (n = 0; n < N; ++n)
				sumGamma += gamma(n, k);
			pi_[k] = 0.001 + 0.999 * sumGamma / N;

			// reestimate observation(emission) distribution in each state.
			doEstimateObservationDensityParametersByML(N, (unsigned int)k, observations, gamma, sumGamma);
		}

		// E-step: evaluate gamma.
		computeGamma(N, observations, gamma, logProb);

		// compute difference between log probability of two iterations.
#if 1
		delta = logProb - finalLogProbability;
#else
		delta = std::fabs(logProb - finalLogProbability);
#endif

		finalLogProbability = logProb;  // log P(observations | estimated model).
		++numIteration;
	} while (delta > terminationTolerance && numIteration <= maxIteration);  // if log probability does not change much, exit.

	return true;
}

bool ContinuousDensityMixtureModel::trainByMAPUsingConjugatePrior(const std::size_t N, const dmatrix_type &observations, const double terminationTolerance, const std::size_t maxIteration, std::size_t &numIteration, double &initLogProbability, double &finalLogProbability)
{
	//	[ref] "Maximum a Posteriori Estimation for Multivariate Gaussian Mixture Observations of Markov Chains", J.-L. Gauvain adn C.-H. Lee, TSAP, 1994.

	if (!doDoHyperparametersOfConjugatePriorExist())
		throw std::runtime_error("Hyperparameters of the conjugate prior have to be assigned for MAP learning.");

	dmatrix_type gamma(N, K_, 0.0);
	double logProb;

	// E-step: evaluate gamma.
	computeGamma(N, observations, gamma, logProb);

	initLogProbability = logProb;  // log P(observations | initial model).
	finalLogProbability = logProb;

	std::size_t n, k;
	double denominatorPhi = double(N) - double(K_);
	for (k = 0; k < K_; ++k)
		denominatorPhi += (*pi_conj_)[k];
	const double factorPhi = 0.999 / denominatorPhi;

	double delta;
	numIteration = 0;
	double sumGamma;
	do
	{
		// M-step.
		for (k = 0; k < K_; ++k)
		{
			sumGamma = 0.0;
			for (n = 0; n < N; ++n)
				sumGamma += gamma(n, k);

			// reestimate the mixture coefficient of state k.
			pi_[k] = 0.001 + (sumGamma + (*pi_conj_)[k] - 1.0) * factorPhi;

			// reestimate observation(emission) distribution in each state.
			doEstimateObservationDensityParametersByMAPUsingConjugatePrior(N, (unsigned int)k, observations, gamma, sumGamma);
		}

		// E-step: evaluate gamma.
		computeGamma(N, observations, gamma, logProb);

		// compute difference between log probability of two iterations.
#if 1
		delta = logProb - finalLogProbability;
#else
		delta = std::fabs(logProb - finalLogProbability);
#endif

		finalLogProbability = logProb;  // log P(observations | estimated model).
		++numIteration;
	} while (delta > terminationTolerance && numIteration <= maxIteration);  // if log probability does not change much, exit.

	return true;
}

bool ContinuousDensityMixtureModel::trainByMAPUsingEntropicPrior(const std::size_t N, const dmatrix_type &observations, const double z, const bool doesTrimParameter, const double terminationTolerance, const std::size_t maxIteration, std::size_t &numIteration, double &initLogProbability, double &finalLogProbability)
{
	// [ref] "Structure Learning in Conditional Probability Models via an Entropic Prior and Parameter Extinction", M. Brand, Neural Computation, 1999.

	//if (!doDoHyperparametersOfEntropicPriorExist())
	//	throw std::runtime_error("Hyperparameters of the entropic prior have to be assigned for MAP learning.");

	dmatrix_type gamma(N, K_, 0.0);
	double logProb;
	const double eps = 1e-50;

	// E-step: evaluate gamma.
	computeGamma(N, observations, gamma, logProb);

	initLogProbability = logProb;  // log P(observations | initial model).
	finalLogProbability = logProb;

	std::vector<double> omega(K_, 0.0), theta(K_, 0.0);
	dmatrix_type prob(doesTrimParameter ? N : 0, doesTrimParameter ? K_ : 0, 0.0);
	double numerator, denominator;
	double delta;
	std::size_t n, k, i;
	numIteration = 0;
	double sumGamma, sumTheta;
	bool isTrimmed;
	do
	{
		// M-step.
		// reestimate mixture coefficients(weights).
		for (k = 0; k < K_; ++k)
		{
			omega[k] = 0.0;
			for (n = 0; n < N; ++n)
				omega[k] += gamma(n, k);
		}

		double entropicMAPLogLikelihood = 0.0;
		const bool retval = computeMAPEstimateOfMultinomialUsingEntropicPrior(omega, z, theta, entropicMAPLogLikelihood, terminationTolerance, maxIteration, false);
		assert(retval);

		// trim mixture coefficients(weights).
		if (doesTrimParameter)
		{
			for (n = 0; n < N; ++n)
			{
				const boost::numeric::ublas::matrix_row<const dmatrix_type> obs(observations, n);
				for (k = 0; k < K_; ++k)
					prob(n, k) = doEvaluateMixtureComponentProbability(k, obs);
			}

			isTrimmed = false;
			for (k = 0; k < K_; ++k)
			{
				numerator = 0.0;
				denominator = 0.0;
				for (n = 0; n < N; ++n)
				{
					numerator += prob(n, k);
					for (i = 0; i < K_; ++i)
						denominator += prob(n, i) * theta[i];
				}

				if (theta[k] <= std::exp(-numerator / denominator))
				{
					theta[k] = 0.0;
					isTrimmed = true;
				}
			}

			if (isTrimmed)
			{
				sumTheta = std::accumulate(theta.begin(), theta.end(), 0.0);
				assert(std::fabs(sumTheta) >= eps);
				for (k = 0; k < K_; ++k)
					theta[k] /= sumTheta;
			}
		}

		pi_.assign(theta.begin(), theta.end());

		// reestimate observation(emission) distribution in each state.
		for (k = 0; k < K_; ++k)
		{
			sumGamma = 0.0;
			for (n = 0; n < N; ++n)
				sumGamma += gamma(n, k);

			doEstimateObservationDensityParametersByMAPUsingEntropicPrior(N, (unsigned int)k, observations, gamma, z, doesTrimParameter, sumGamma);
		}

		// E-step: evaluate gamma.
		computeGamma(N, observations, gamma, logProb);

		// compute difference between log probability of two iterations.
#if 1
		delta = logProb - finalLogProbability;
#else
		delta = std::fabs(logProb - finalLogProbability);
#endif

		finalLogProbability = logProb;  // log P(observations | estimated model).
		++numIteration;
	} while (delta > terminationTolerance && numIteration <= maxIteration);  // if log probability does not change much, exit.

	return true;
}

void ContinuousDensityMixtureModel::generateSample(const std::size_t N, dmatrix_type &observations, std::vector<unsigned int> &states, const unsigned int seed /*= (unsigned int)-1*/) const
{
	// PRECONDITIONS [] >>
	//	-. std::srand() has to be called before this function is called.

	doInitializeRandomSampleGeneration(seed);

	for (std::size_t n = 0; n < N; ++n)
	{
		states[n] = generateState();
#if defined(__GNUC__)
		boost::numeric::ublas::matrix_row<dmatrix_type> obs(observations, n);
		doGenerateObservationsSymbol(states[n], obs, seed);
#else
		doGenerateObservationsSymbol(states[n], boost::numeric::ublas::matrix_row<dmatrix_type>(observations, n));
#endif
	}

	doFinalizeRandomSampleGeneration();
}

double ContinuousDensityMixtureModel::evaluateEmissionProbability(const boost::numeric::ublas::matrix_row<const dmatrix_type> &observation) const
{
	double prob = 0.0;
	for (size_t k = 0; k < K_; ++k)
		prob += pi_[k] * doEvaluateMixtureComponentProbability(k, observation);

	return prob;
}

/*static*/ bool ContinuousDensityMixtureModel::readSequence(std::istream &stream, std::size_t &N, std::size_t &D, dmatrix_type &observations)
{
	std::string dummy;

	stream >> dummy >> N;
#if defined(__GNUC__)
	if (strcasecmp(dummy.c_str(), "N=") != 0)
#elif defined(_MSC_VER)
	if (_stricmp(dummy.c_str(), "N=") != 0)
#endif
		return false;

	stream >> dummy >> D;
#if defined(__GNUC__)
	if (strcasecmp(dummy.c_str(), "D=") != 0)
#elif defined(_MSC_VER)
	if (_stricmp(dummy.c_str(), "D=") != 0)
#endif
		return false;

	observations.resize(N, D);
	for (std::size_t n = 0; n < N; ++n)
		for (std::size_t i = 0; i < D; ++i)
			stream >> observations(n, i);

	return true;
}

/*static*/ bool ContinuousDensityMixtureModel::writeSequence(std::ostream &stream, const dmatrix_type &observations)
{
	const std::size_t N = observations.size1();
	const std::size_t D = observations.size2();

	stream << "N= " << N << std::endl;
	stream << "D= " << D << std::endl;
	for (std::size_t n = 0; n < N; ++n)
	{
		for (std::size_t i = 0; i < D; ++i)
			stream << observations(n, i) << ' ';
		stream << std::endl;
	}

	return true;
}

}  // namespace swl
