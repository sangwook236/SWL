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
	double numeratorPr, denominatorPr;
	double delta, logprobprev = logprobf;
	size_t i, k, n;
	size_t iter = 0;
	do
	{
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

			// for multivariate normal distributions
			// TODO [check] >> this code may be changed into a vector form.
			for (i = 0; i < D_; ++i)
			{
				numeratorPr = 0.0;
				for (n = 0; n < N; ++n)
					numeratorPr += gamma[n][k] * observations[n][i];
				mus_[k][i] = 0.001 + 0.999 * numeratorPr / denominatorPr;
			}

			// for multivariate normal distributions
			// FIXME [modify] >> this code may be changed into a matrix form.
			throw std::runtime_error("this code may be changed into a matrix form.");
/*
			boost::multi_array<double, 3>::array_view<2>::type sigma = sigmas_[boost::indices[k][boost::multi_array<double, 3>::index_range()][boost::multi_array<double, 3>::index_range()]];
			for (i = 0; i < D_; ++i)
			{
				numeratorPr = 0.0;
				for (n = 0; n < N; ++n)
					numeratorPr += gamma[n][k] * (observations[n][i] - mus_[k][i]) * (observations[n][i] - mus_[k][i]).tranpose();
				sigma = 0.001 + 0.999 * numeratorPr / denominatorPr;
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

bool HmmWithMultivariateGaussianObservations::estimateParameters(const std::vector<size_t> &Ns, const std::vector<boost::multi_array<double, 2> > &observationSequences, const double terminationTolerance, size_t &numIteration,std::vector<double> &initLogProbabilities, std::vector<double> &finalLogProbabilities)
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

			// for multivariate normal distributions
			// TODO [check] >> this code may be changed into a vector form.
			for (i = 0; i < D_; ++i)
			{
				numeratorPr = 0.0;
				for (r = 0; r < R; ++r)
					for (n = 0; n < Ns[r]; ++n)
						numeratorPr += gammas[r][n][k] * observationSequences[r][n][i];
				mus_[k][i] = 0.001 + 0.999 * numeratorPr / denominatorPr;
			}

			// for multivariate normal distributions
			// FIXME [modify] >> this code may be changed into a matrix form.
			throw std::runtime_error("this code may be changed into a matrix form.");
/*
			boost::multi_array<double, 3>::array_view<2>::type sigma = sigmas_[boost::indices[k][boost::multi_array<double, 3>::index_range()][boost::multi_array<double, 3>::index_range()]];
			for (i = 0; i < D_; ++i)
			{
				numeratorPr = 0.0;
				for (r = 0; r < R; ++r)
					for (n = 0; n < N; ++n)
						numeratorPr += gammas[r][n][k] * (observationSequences[r][n][i] - mus_[k][i]) * (observationSequences[r][n][i] - mus_[k][i]).tranpose();
				sigma = 0.001 + 0.999 * numeratorPr / denominatorPr;
			}
*/
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

	return true;
}

double HmmWithMultivariateGaussianObservations::evaluateEmissionProbability(const unsigned int state, const boost::multi_array<double, 2>::const_array_view<1>::type &observation) const
{
	throw std::runtime_error("not yet implemented");
}

void HmmWithMultivariateGaussianObservations::generateObservationsSymbol(const unsigned int state, boost::multi_array<double, 2>::array_view<1>::type &observation, const unsigned int seed /*= (unsigned int)-1*/) const
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
