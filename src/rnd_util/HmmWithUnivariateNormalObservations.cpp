#include "swl/Config.h"
#include "swl/rnd_util/HmmWithUnivariateNormalObservations.h"
#include <boost/math/distributions/normal.hpp>  // for normal distribution
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <ctime>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

HmmWithUnivariateNormalObservations::HmmWithUnivariateNormalObservations(const size_t K)
: base_type(K, 1), mus_(K, 0.0), sigmas_(K, 0.0),  // 0-based index
  baseGenerator_()
{
}

HmmWithUnivariateNormalObservations::HmmWithUnivariateNormalObservations(const size_t K, const std::vector<double> &pi, const boost::multi_array<double, 2> &A, const std::vector<double> &mus, const std::vector<double> &sigmas)
: base_type(K, 1, pi, A), mus_(mus), sigmas_(sigmas),
  baseGenerator_()
{
}

HmmWithUnivariateNormalObservations::~HmmWithUnivariateNormalObservations()
{
}

bool HmmWithUnivariateNormalObservations::estimateParameters(const size_t N, const boost::multi_array<double, 2> &observations, const double terminationTolerance, boost::multi_array<double, 2> &alpha, boost::multi_array<double, 2> &beta, boost::multi_array<double, 2> &gamma, size_t &numIteration, double &initLogProbability, double &finalLogProbability)
{
	std::vector<double> scale(N, 0.0);
	double logprobf, logprobb;

	// E-step
	runForwardAlgorithm(N, observations, scale, alpha, logprobf);
	runBackwardAlgorithm(N, observations, scale, beta, logprobb);

	computeGamma(N, alpha, beta, gamma);
	boost::multi_array<double, 3> xi(boost::extents[N][K_][K_]);
	computeXi(N, observations, alpha, beta, xi);

	initLogProbability = logprobf;  // log P(observations | initial model)
	finalLogProbability = logprobf;

	double numeratorA, denominatorA;
	double numeratorPr, denominatorPr;
	double delta;
	size_t i, k, n;
	numIteration = 0;
	do
	{
		// M-step
		for (k = 0; k < K_; ++k)
		{
			// reestimate frequency of state k in time n=0
			pi_[k] = 0.001 + 0.999 * gamma[0][k];

			// reestimate transition matrix 
			denominatorA = 0.0;
			for (n = 0; n < N - 1; ++n)
				denominatorA += gamma[n][k];

			for (i = 0; i < K_; ++i)
			{
				numeratorA = 0.0;
				for (n = 0; n < N - 1; ++n)
					numeratorA += xi[n][k][i];
				A_[k][i] = 0.001 + 0.999 * numeratorA / denominatorA;
			}

			// reestimate symbol prob in each state
			denominatorPr = denominatorA + gamma[N-1][k];

			// for univariate normal distributions
			numeratorPr = 0.0;
			for (n = 0; n < N; ++n)
				numeratorPr += gamma[n][k] * observations[n][0];
			mus_[k] = 0.001 + 0.999 * numeratorPr / denominatorPr;

			// for univariate normal distributions
			numeratorPr = 0.0;
			for (n = 0; n < N; ++n)
				numeratorPr += gamma[n][k] * (observations[n][0] - mus_[k]) * (observations[n][0] - mus_[k]);
			sigmas_[k] = 0.001 + 0.999 * numeratorPr / denominatorPr;
		}

		// E-step
		runForwardAlgorithm(N, observations, scale, alpha, logprobf);
		runBackwardAlgorithm(N, observations, scale, beta, logprobb);

		computeGamma(N, alpha, beta, gamma);
		computeXi(N, observations, alpha, beta, xi);

		// compute difference between log probability of two iterations
#if 1
		delta = logprobf - finalLogProbability;
#else
		delta = std::fabs(logprobf - finalLogProbability);  // log P(observations | estimated model)
#endif

		finalLogProbability = logprobf;  // log P(observations | estimated model)
		++numIteration;
	} while (delta > terminationTolerance);  // if log probability does not change much, exit

	return true;
}

bool HmmWithUnivariateNormalObservations::estimateParameters(const std::vector<size_t> &Ns, const std::vector<boost::multi_array<double, 2> > &observationSequences, const double terminationTolerance, size_t &numIteration,std::vector<double> &initLogProbabilities, std::vector<double> &finalLogProbabilities)
{
	const size_t R = Ns.size();  // number of observations sequences
	size_t Nr, r;

	std::vector<boost::multi_array<double, 2> > alphas, betas, gammas;
	std::vector<boost::multi_array<double, 3> > xis;
	std::vector<std::vector<double> > scales;
	alphas.reserve(R);
	betas.reserve(R);
	gammas.reserve(R);
	xis.reserve(R);
	scales.reserve(R);
	for (r = 0; r < R; ++r)
	{
		Nr = Ns[r];
		alphas.push_back(boost::multi_array<double, 2>(boost::extents[Nr][K_]));
		betas.push_back(boost::multi_array<double, 2>(boost::extents[Nr][K_]));
		gammas.push_back(boost::multi_array<double, 2>(boost::extents[Nr][K_]));
		xis.push_back(boost::multi_array<double, 3>(boost::extents[Nr][K_][K_]));
		scales.push_back(std::vector<double>(Nr, 0.0));
	}

	double logprobf, logprobb;

	// E-step
	for (r = 0; r < R; ++r)
	{
		Nr = Ns[r];
		const boost::multi_array<double, 2> &observations = observationSequences[r];

		boost::multi_array<double, 2> &alphar = alphas[r];
		boost::multi_array<double, 2> &betar = betas[r];
		boost::multi_array<double, 2> &gammar = gammas[r];
		boost::multi_array<double, 3> &xir = xis[r];
		std::vector<double> &scaler = scales[r];

		runForwardAlgorithm(Nr, observations, scaler, alphar, logprobf);
		runBackwardAlgorithm(Nr, observations, scaler, betar, logprobb);

		computeGamma(Nr, alphar, betar, gammar);
		computeXi(Nr, observations, alphar, betar, xir);

		initLogProbabilities[r] = logprobf;  // log P(observations | initial model)
		finalLogProbabilities[r] = logprobf;
	}

	double numeratorPi;
	double numeratorA, denominatorA;
	double numeratorPr, denominatorPr;
	double delta;;
	bool continueToLoop;
	size_t i, k, n;
	numIteration = 0;
	do
	{
		// M-step
		for (k = 0; k < K_; ++k)
		{
			// reestimate frequency of state k in time n=0
			numeratorPi = 0.0;
			for (r = 0; r < R; ++r)
				numeratorPi += gammas[r][0][k];
			pi_[k] = 0.001 + 0.999 * numeratorPi / (double)R;

			// reestimate transition matrix 
			denominatorA = 0.0;
			for (r = 0; r < R; ++r)
				for (n = 0; n < Ns[r] - 1; ++n)
					denominatorA += gammas[r][n][k];

			for (i = 0; i < K_; ++i)
			{
				numeratorA = 0.0;
				for (r = 0; r < R; ++r)
					for (n = 0; n < Ns[r] - 1; ++n)
						numeratorA += xis[r][n][k][i];
				A_[k][i] = 0.001 + 0.999 * numeratorA / denominatorA;
			}

			// reestimate symbol prob in each state
			denominatorPr = denominatorA;
			for (r = 0; r < R; ++r)
				denominatorPr += gammas[r][Ns[r]-1][k];

			// for univariate normal distributions
			numeratorPr = 0.0;
			for (r = 0; r < R; ++r)
				for (n = 0; n < Ns[r]; ++n)
					numeratorPr += gammas[r][n][k] * observationSequences[r][n][0];
			mus_[k] = 0.001 + 0.999 * numeratorPr / denominatorPr;

			// for univariate normal distributions
			numeratorPr = 0.0;
			for (r = 0; r < R; ++r)
				for (n = 0; n < Ns[r]; ++n)
					numeratorPr += gammas[r][n][k] * (observationSequences[r][n][0] - mus_[k]) * (observationSequences[r][n][0] - mus_[k]);
			sigmas_[k] = 0.001 + 0.999 * numeratorPr / denominatorPr;
		}

		// E-step
		continueToLoop = false;
		for (r = 0; r < R; ++r)
		{
			Nr = Ns[r];
			const boost::multi_array<double, 2> &observations = observationSequences[r];

			boost::multi_array<double, 2> &alphar = alphas[r];
			boost::multi_array<double, 2> &betar = betas[r];
			boost::multi_array<double, 2> &gammar = gammas[r];
			boost::multi_array<double, 3> &xir = xis[r];
			std::vector<double> &scaler = scales[r];

			runForwardAlgorithm(Nr, observations, scaler, alphar, logprobf);
			runBackwardAlgorithm(Nr, observations, scaler, betar, logprobb);

			computeGamma(Nr, alphar, betar, gammar);
			computeXi(Nr, observations, alphar, betar, xir);

			// compute difference between log probability of two iterations
#if 1
			delta = logprobf - finalLogProbabilities[r];
#else
			delta = std::fabs(logprobf - finalLogProbabilities[r]);
#endif
			if (delta > terminationTolerance)
				continueToLoop = true;

			finalLogProbabilities[r] = logprobf;  // log P(observations | estimated model)
		}

		++numIteration;
	} while (continueToLoop);  // if log probability does not change much, exit

	// compute gamma & xi
/*
	{
		// gamma can use the result from Baum-Welch algorithm
		//boost::multi_array<double, 2> gamma2(boost::extents[Nr][K_]);
		//cdhmm->computeGamma(Nr, alphar, betar, gamma2);

		//
		boost::multi_array<double, 3> xi2(boost::extents[Nr][K_][K_]);
		cdhmm->computeXi(Nr, observations, alphar, betar, xi2);
	}
*/

	return true;
}

double HmmWithUnivariateNormalObservations::evaluateEmissionProbability(const unsigned int state, const boost::multi_array<double, 2>::const_array_view<1>::type &observation) const
{
	//boost::math::normal pdf;  // (default mean = zero, and standard deviation = unity)
	boost::math::normal pdf(mus_[state], sigmas_[state]);

	return boost::math::pdf(pdf, observation[0]);
}

void HmmWithUnivariateNormalObservations::generateObservationsSymbol(const unsigned int state, boost::multi_array<double, 2>::array_view<1>::type &observation, const unsigned int seed /*= (unsigned int)-1*/) const
{
	typedef boost::normal_distribution<> distribution_type;
	typedef boost::variate_generator<base_generator_type &, distribution_type> generator_type;

	if ((unsigned int)-1 != seed)
		baseGenerator_.seed(seed);

	generator_type normal_gen(baseGenerator_, distribution_type(mus_[state], sigmas_[state]));
	for (size_t i = 0; i < D_; ++i)
		observation[i] = normal_gen();
}

bool HmmWithUnivariateNormalObservations::readObservationDensity(std::istream &stream)
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

	stream >> dummy;
#if defined(__GNUC__)
	if (strcasecmp(dummy.c_str(), "mu:") != 0)
#elif defined(_MSC_VER)
	if (_stricmp(dummy.c_str(), "mu:") != 0)
#endif
		return false;

	for (size_t k = 0; k < K_; ++k)
		stream >> mus_[k];

	stream >> dummy;
#if defined(__GNUC__)
	if (strcasecmp(dummy.c_str(), "sigma:") != 0)
#elif defined(_MSC_VER)
	if (_stricmp(dummy.c_str(), "sigma:") != 0)
#endif
		return false;

	for (size_t k = 0; k < K_; ++k)
		stream >> sigmas_[k];

	return true;
}

bool HmmWithUnivariateNormalObservations::writeObservationDensity(std::ostream &stream) const
{
	stream << "univariate normal:" << std::endl;

	stream << "mu:" << std::endl;
	for (size_t k = 0; k < K_; ++k)
		stream << mus_[k] << ' ';
	stream << std::endl;
	
	stream << "sigma:" << std::endl;
	for (size_t k = 0; k < K_; ++k)
		stream << sigmas_[k] << ' ';
	stream << std::endl;

	return true;
}

void HmmWithUnivariateNormalObservations::initializeObservationDensity()
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
