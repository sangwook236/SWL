#include "swl/Config.h"
#include "swl/rnd_util/CDHMM.h"
#include "RndUtilLocalApi.h"
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <numeric>
#include <cstring>
#include <stdexcept>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

CDHMM::CDHMM(const size_t K, const size_t D)
: base_type(K, D)
{
}

CDHMM::CDHMM(const size_t K, const size_t D, const dvector_type &pi, const dmatrix_type &A)
: base_type(K, D, pi, A)
{
}

CDHMM::CDHMM(const size_t K, const size_t D, const dvector_type *pi_conj, const dmatrix_type *A_conj)
: base_type(K, D, pi_conj, A_conj)
{
}

CDHMM::~CDHMM()
{
}

void CDHMM::runForwardAlgorithm(const size_t N, const dmatrix_type &observations, dmatrix_type &alpha, double &likelihood) const
{
	size_t i, k;  // state indices

	// 1. Initialization
	for (k = 0; k < K_; ++k)
		//alpha(0, k) = pi_[k] * B_(k, observations[0]);
		//alpha(0, k) = pi_[k] * doEvaluateEmissionProbability(k, boost::numeric::ublas::matrix_row<const dmatrix_type>(observations, 0));
		alpha(0, k) = pi_[k] * doEvaluateEmissionProbability(k, 0, observations);

	// 2. Induction
	double sum;  // partial sum
	size_t n_1;
	for (size_t n = 1; n < N; ++n)
	{
		n_1 = n - 1;
		for (k = 0; k < K_; ++k)
		{
			sum = 0.0;
			for (i = 0; i < K_; ++i)
				sum += alpha(n_1, i) * A_(i, k);

			//alpha(n, k) = sum * B_(k, observations[n]);
			//alpha(n, k) = sum * doEvaluateEmissionProbability(k, boost::numeric::ublas::matrix_row<const dmatrix_type>(observations, n));
			alpha(n, k) = sum * doEvaluateEmissionProbability(k, n, observations);
		}
	}

	// 3. Termination
	likelihood = 0.0;
	n_1 = N - 1;
	for (k = 0; k < K_; ++k)
		likelihood += alpha(n_1, k);
}

void CDHMM::runForwardAlgorithm(const size_t N, const dmatrix_type &observations, dvector_type &scale, dmatrix_type &alpha, double &logLikelihood) const
{
	size_t i, k;  // state indices

	// 1. Initialization
	scale[0] = 0.0;
	for (k = 0; k < K_; ++k)
	{
		//alpha(0, k) = pi_[k] * B_(k, observations[0]);
		//alpha(0, k) = pi_[k] * doEvaluateEmissionProbability(k, boost::numeric::ublas::matrix_row<const dmatrix_type>(observations, 0));
		alpha(0, k) = pi_[k] * doEvaluateEmissionProbability(k, 0, observations);
		scale[0] += alpha(0, k);
	}
	for (k = 0; k < K_; ++k)
		alpha(0, k) /= scale[0];

	// 2. Induction
	double sum;  // partial sum
	size_t n, n_1;
	for (n = 1; n < N; ++n)
	{
		n_1 = n - 1;
		scale[n] = 0.0;
		for (k = 0; k < K_; ++k)
		{
			sum = 0.0;
			for (i = 0; i < K_; ++i)
				sum += alpha(n_1, i) * A_(i, k);

			//alpha(n, k) = sum * B_(k, observations[n]);
			//alpha(n, k) = sum * doEvaluateEmissionProbability(k, boost::numeric::ublas::matrix_row<const dmatrix_type>((dmatrix_type)observations, n));
			alpha(n, k) = sum * doEvaluateEmissionProbability(k, n, observations);
			scale[n] += alpha(n, k);
		}
		for (k = 0; k < K_; ++k)
			alpha(n, k) /= scale[n];
	}

	// 3. Termination
	logLikelihood = 0.0;
	for (n = 0; n < N; ++n)
		logLikelihood += std::log(scale[n]);
}

void CDHMM::runBackwardAlgorithm(const size_t N, const dmatrix_type &observations, dmatrix_type &beta) const
{
	size_t i, k;  // state indices
	size_t n_1;

	// 1. Initialization
	n_1 = N - 1;
	for (k = 0; k < K_; ++k)
		beta(n_1, k) = 1.0;

	// 2. Induction
	double sum;
	for (size_t n = N - 1; n > 0; --n)
	{
		n_1 = n - 1;
		for (k = 0; k < K_; ++k)
		{
			sum = 0.0;
			for (i = 0; i < K_; ++i)
				//sum += A_(k, i) * B_(i, observations[n]) * beta(n, i);
				//sum += A_(k, i) * doEvaluateEmissionProbability(i, boost::numeric::ublas::matrix_row<const dmatrix_type>(observations, n)) * beta(n, i);
				sum += A_(k, i) * doEvaluateEmissionProbability(i, n, observations) * beta(n, i);
			beta(n_1, k) = sum;
		}
	}
/*
	// 3. Termination
	probability = 0.0;
	for (k = 0; k < K_; ++k)
		probability += beta(0, k);
*/
}

void CDHMM::runBackwardAlgorithm(const size_t N, const dmatrix_type &observations, const dvector_type &scale, dmatrix_type &beta) const
{
	size_t i, k;  // state indices
	size_t n_1;

	// 1. Initialization
	n_1 = N - 1;
	for (k = 0; k < K_; ++k)
		beta(n_1, k) = 1.0 / scale[n_1];

	// 2. Induction
	double sum;
	for (size_t n = N - 1; n > 0; --n)
	{
		n_1 = n - 1;
		for (k = 0; k < K_; ++k)
		{
			sum = 0.0;
			for (i = 0; i < K_; ++i)
				//sum += A_(k, i) * B_(i, observations[n]) * beta(n, i);
				//sum += A_(k, i) * doEvaluateEmissionProbability(i, boost::numeric::ublas::matrix_row<const dmatrix_type>(observations, n)) * beta(n, i);
				sum += A_(k, i) * doEvaluateEmissionProbability(i, n, observations) * beta(n, i);
			beta(n_1, k) = sum / scale[n];
		}
	}
}

void CDHMM::runViterbiAlgorithm(const size_t N, const dmatrix_type &observations, dmatrix_type &delta, uimatrix_type &psi, uivector_type &states, double &probability, const bool useLog /*= true*/) const
{
	if (useLog) runViterbiAlgorithmUsingLog(N, observations, delta, psi, states, probability);
	else runViterbiAlgorithmNotUsigLog(N, observations, delta, psi, states, probability);
}

void CDHMM::runViterbiAlgorithmNotUsigLog(const size_t N, const dmatrix_type &observations, dmatrix_type &delta, uimatrix_type &psi, uivector_type &states, double &probability) const
{
	size_t i, k;  // state indices

	// 1. Initialization
	for (k = 0; k < K_; ++k)
	{
		//delta(0, k) = pi_[k] * B_(k, observations[0]);
		//delta(0, k) = pi_[k] * doEvaluateEmissionProbability(k, boost::numeric::ublas::matrix_row<const dmatrix_type>(observations, 0));
		delta(0, k) = pi_[k] * doEvaluateEmissionProbability(k, 0, observations);
		psi(0, k) = 0u;
	}

	// 2. Recursion
	size_t maxvalind;
	double maxval, val;
	size_t n, n_1;
	for (n = 1; n < N; ++n)
	{
		n_1 = n - 1;
		for (k = 0; k < K_; ++k)
		{
			maxval = 0.0;
			maxvalind = 0;
			for (i = 0; i < K_; ++i)
			{
				val = delta(n_1, i) * A_(i, k);
				if (val > maxval)
				{
					maxval = val;
					maxvalind = i;
				}
			}

			//delta(n, k) = maxval * B_(k, observations[n]);
			//delta(n, k) = maxval * doEvaluateEmissionProbability(k, boost::numeric::ublas::matrix_row<const dmatrix_type>(observations, n));
			delta(n, k) = maxval * doEvaluateEmissionProbability(k, n, observations);
			psi(n, k) = (unsigned int)maxvalind;
		}
	}

	// 3. Termination
	probability = 0.0;
	n_1 = N - 1;
	states[n_1] = 0u;
	for (k = 0; k < K_; ++k)
	{
		if (delta(n_1, k) > probability)
		{
			probability = delta(n_1, k);
			states[n_1] = (unsigned int)k;
		}
	}

	// 4. Path (state sequence) backtracking
	for (n = N - 1; n > 0; --n)
		states[n-1] = psi(n, states[n]);
}

void CDHMM::runViterbiAlgorithmUsingLog(const size_t N, const dmatrix_type &observations, dmatrix_type &delta, uimatrix_type &psi, uivector_type &states, double &probability) const
{
	size_t i, k;  // state indices
	size_t n;

	// 0. Preprocessing
	dvector_type logPi(pi_);
	dmatrix_type logA(A_);
	dmatrix_type logO(K_, N);
	for (k = 0; k < K_; ++k)
	{
		logPi[k] = std::log(pi_[k]);

		for (i = 0; i < K_; ++i)
			logA(k, i) = std::log(A_(k, i));

		for (n = 0; n < N; ++n)
			//logO(k, n) = std::log(B_(k, observations[n]));
			//logO(k, n) = std::log(doEvaluateEmissionProbability(k, boost::numeric::ublas::matrix_row<const dmatrix_type>(observations, n)));
			logO(k, n) = std::log(doEvaluateEmissionProbability(k, n, observations));
	}

	// 1. Initialization
	for (k = 0; k < K_; ++k)
	{
		delta(0, k) = logPi[k] + logO(k, 0);
		psi(0, k) = 0u;
	}

	// 2. Recursion
	size_t maxvalind;
	double maxval, val;
	size_t n_1;
	for (n = 1; n < N; ++n)
	{
		n_1 = n - 1;
		for (k = 0; k < K_; ++k)
		{
			maxval = -std::numeric_limits<double>::max();
			maxvalind = 0;
			for (i = 0; i < K_; ++i)
			{
				val = delta(n_1, i) + logA(i, k);
				if (val > maxval)
				{
					maxval = val;
					maxvalind = i;
				}
			}

			delta(n, k) = maxval + logO(k, n);
			psi(n, k) = (unsigned int)maxvalind;
		}
	}

	// 3. Termination
	probability = -std::numeric_limits<double>::max();
	n_1 = N - 1;
	states[n_1] = 0u;
	for (k = 0; k < K_; ++k)
	{
		if (delta(n_1, k) > probability)
		{
			probability = delta(n_1, k);
			states[n_1] = (unsigned int)k;
		}
	}

	// 4. Path (state sequence) backtracking
	for (n = N - 1; n > 0; --n)
		states[n-1] = psi(n, states[n]);
}

bool CDHMM::trainByML(const size_t N, const dmatrix_type &observations, const double terminationTolerance, const size_t maxIteration, size_t &numIteration, double &initLogLikelihood, double &finalLogLikelihood)
{
	dvector_type scale(N, 0.0);
	double logLikelihood;
	size_t n;

	dmatrix_type alpha(N, K_, 0.0), beta(N, K_, 0.0), gamma(N, K_, 0.0);
	std::vector<dmatrix_type> xi;
	xi.reserve(N - 1);
	for (n = 0; n < N - 1; ++n)
		xi.push_back(dmatrix_type(K_, K_, 0.0));

	// E-step: evaluate gamma & xi.
	{
		// forward-backward algorithm.
		runForwardAlgorithm(N, observations, scale, alpha, logLikelihood);
		runBackwardAlgorithm(N, observations, scale, beta);

		computeGamma(N, alpha, beta, gamma);
		computeXi(N, observations, alpha, beta, xi);
	}

	initLogLikelihood = logLikelihood;  // log P(observations | initial model).
	finalLogLikelihood = logLikelihood;

	//
	double numeratorA, denominatorA;
	double delta;
	size_t i, k;
	numIteration = 0;
	do
	{
		// M-step.
		for (k = 0; k < K_; ++k)
		{
			// reestimate frequency of state k in time n=0.
			pi_[k] = 0.001 + 0.999 * gamma(0, k);

			// reestimate transition matrix.
			denominatorA = 0.0;
			for (n = 0; n < N - 1; ++n)
				denominatorA += gamma(n, k);

			for (i = 0; i < K_; ++i)
			{
				numeratorA = 0.0;
				for (n = 0; n < N - 1; ++n)
					numeratorA += xi[n](k, i);
				A_(k, i) = 0.001 + 0.999 * numeratorA / denominatorA;
			}

			// reestimate observation(emission) distribution in each state.
			// run E-step & M-step as well.
			doEstimateObservationDensityParametersByML(N, (unsigned int)k, observations, gamma, denominatorA);
		}

		// E-step: evaluate gamma & xi.
		{
			// forward-backward algorithm
			runForwardAlgorithm(N, observations, scale, alpha, logLikelihood);
			runBackwardAlgorithm(N, observations, scale, beta);

			computeGamma(N, alpha, beta, gamma);
			computeXi(N, observations, alpha, beta, xi);
		}

		// compute difference between log probability of two iterations.
#if 1
		delta = logLikelihood - finalLogLikelihood;
#else
		delta = std::fabs(logLikelihood - finalLogLikelihood);
#endif

		finalLogLikelihood = logLikelihood;  // log P(observations | estimated model).
		++numIteration;
	} while (delta > terminationTolerance && numIteration <= maxIteration);  // if log probability does not change much, exit.

/*
	// compute gamma & xi.
	{
		// gamma can use the result from Baum-Welch algorithm.
		//dmatrix_type gamma2(boost::extents[N][K_]);
		//computeGamma(N, alpha, beta, gamma2);

		//
		std::vector<dmatrix_type> xi2;
		xi2.reserve(N - 1);
		for (n = 0; n < N - 1; ++n)
			xi2.push_back(dmatrix_type(K_, K_, 0.0));
		computeXi(N, observations, alpha, beta, xi2);
	}
*/

	return true;
}

bool CDHMM::trainByML(const std::vector<size_t> &Ns, const std::vector<dmatrix_type> &observationSequences, const double terminationTolerance, const size_t maxIteration, size_t &numIteration, std::vector<double> &initLogLikelihoods, std::vector<double> &finalLogLikelihoods)
{
	const size_t R = Ns.size();  // number of observations sequences.
	size_t Nr, r, n;

	std::vector<dmatrix_type> alphas, betas, gammas;
	std::vector<std::vector<dmatrix_type> > xis;
	std::vector<dvector_type> scales;
	alphas.reserve(R);
	betas.reserve(R);
	gammas.reserve(R);
	xis.reserve(R);
	scales.reserve(R);
	for (r = 0; r < R; ++r)
	{
		Nr = Ns[r];
		alphas.push_back(dmatrix_type(Nr, K_, 0.0));
		betas.push_back(dmatrix_type(Nr, K_, 0.0));
		gammas.push_back(dmatrix_type(Nr, K_, 0.0));
		xis.push_back(std::vector<dmatrix_type>());
		xis[r].reserve(Nr - 1);
		for (n = 0; n < Nr - 1; ++n)
			xis[r].push_back(dmatrix_type(K_, K_, 0.0));
		scales.push_back(dvector_type(Nr, 0.0));
	}

	double logLikelihood;

	// E-step: evaluate gamma & xi.
	for (r = 0; r < R; ++r)
	{
		Nr = Ns[r];
		const dmatrix_type &observations = observationSequences[r];

		dmatrix_type &alphar = alphas[r];
		dmatrix_type &betar = betas[r];
		dmatrix_type &gammar = gammas[r];
		std::vector<dmatrix_type> &xir = xis[r];
		dvector_type &scaler = scales[r];

		// forward-backward algorithm
		runForwardAlgorithm(Nr, observations, scaler, alphar, logLikelihood);
		runBackwardAlgorithm(Nr, observations, scaler, betar);

		computeGamma(Nr, alphar, betar, gammar);
		computeXi(Nr, observations, alphar, betar, xir);

		initLogLikelihoods[r] = logLikelihood;  // log P(observations | initial model).
		finalLogLikelihoods[r] = logLikelihood;
	}

	//
	double numeratorPi;
	double numeratorA, denominatorA;
	double delta;;
	bool continueToLoop;
	size_t i, k;
	numIteration = 0;
	do
	{
		// M-step.
		for (k = 0; k < K_; ++k)
		{
			// reestimate frequency of state k in time n=0.
			numeratorPi = 0.0;
			for (r = 0; r < R; ++r)
				numeratorPi += gammas[r](0, k);
			pi_[k] = 0.001 + 0.999 * numeratorPi / (double)R;

			// reestimate transition matrix.
			denominatorA = 0.0;
			for (r = 0; r < R; ++r)
				for (n = 0; n < Ns[r] - 1; ++n)
					denominatorA += gammas[r](n, k);

			for (i = 0; i < K_; ++i)
			{
				numeratorA = 0.0;
				for (r = 0; r < R; ++r)
					for (n = 0; n < Ns[r] - 1; ++n)
						numeratorA += xis[r][n](k, i);
				A_(k, i) = 0.001 + 0.999 * numeratorA / denominatorA;
			}

			// reestimate observation(emission) distribution in each state.
			// run E-step & M-step as well.
			doEstimateObservationDensityParametersByML(Ns, (unsigned int)k, observationSequences, gammas, R, denominatorA);
		}

		// E-step: evaluate gamma & xi.
		continueToLoop = false;
		for (r = 0; r < R; ++r)
		{
			Nr = Ns[r];
			const dmatrix_type &observations = observationSequences[r];

			dmatrix_type &alphar = alphas[r];
			dmatrix_type &betar = betas[r];
			dmatrix_type &gammar = gammas[r];
			std::vector<dmatrix_type> &xir = xis[r];
			dvector_type &scaler = scales[r];

			// forward-backward algorithm.
			runForwardAlgorithm(Nr, observations, scaler, alphar, logLikelihood);
			runBackwardAlgorithm(Nr, observations, scaler, betar);

			computeGamma(Nr, alphar, betar, gammar);
			computeXi(Nr, observations, alphar, betar, xir);

			// compute difference between log probability of two iterations.
#if 1
			delta = logLikelihood - finalLogLikelihoods[r];
#else
			delta = std::fabs(logLikelihood - finalLogLikelihoods[r]);
#endif
			if (delta > terminationTolerance && numIteration <= maxIteration)
				continueToLoop = true;

			finalLogLikelihoods[r] = logLikelihood;  // log P(observations | estimated model).
		}

		++numIteration;
	} while (continueToLoop);  // if log probability does not change much, exit.

/*
	// compute gamma & xi.
	{
		for (r = 0; r < R; ++r)
		{
			// gamma can use the result from Baum-Welch algorithm.
			//dmatrix_type gamma2(Ns[r], K_, 0.0);
			//computeGamma(Ns[r], alphas[r], betas[r], gamma2);

			//
			std::vector<std::vector<dmatrix_type> > xis2;
			xis2.reserve(R);
			for (r = 0; r < R; ++r)
			{
				Nr = Ns[r];
				xis2.push_back(std::vector<dmatrix_type>());
				xis2[r].reserve(Nr - 1);
				for (n = 0; n < Nr - 1; ++n)
					xis2[r].push_back(dmatrix_type(K_, K_, 0.0));
			}
			computeXi(Ns[r], observationSequences[r], alphas[r], betas[r], xis2[r]);
		}
	}
*/

	return true;
}

bool CDHMM::trainByMAPUsingConjugatePrior(const size_t N, const dmatrix_type &observations, const double terminationTolerance, const size_t maxIteration, size_t &numIteration, double &initLogLikelihood, double &finalLogLikelihood)
{
	//	[ref] "Maximum a Posteriori Estimation for Multivariate Gaussian Mixture Observations of Markov Chains", J.-L. Gauvain adn C.-H. Lee, TSAP, 1994.

	if (!doDoHyperparametersOfConjugatePriorExist())
		throw std::runtime_error("Hyperparameters of the conjugate prior have to be assigned for MAP learning.");

	dvector_type scale(N, 0.0);
	double logLikelihood;
	size_t n;

	dmatrix_type alpha(N, K_, 0.0), beta(N, K_, 0.0), gamma(N, K_, 0.0);
	std::vector<dmatrix_type> xi;
	xi.reserve(N - 1);
	for (n = 0; n < N - 1; ++n)
		xi.push_back(dmatrix_type(K_, K_, 0.0));

	// E-step: evaluate gamma & xi.
	{
		// forward-backward algorithm.
		runForwardAlgorithm(N, observations, scale, alpha, logLikelihood);
		runBackwardAlgorithm(N, observations, scale, beta);

		computeGamma(N, alpha, beta, gamma);
		computeXi(N, observations, alpha, beta, xi);
	}

	initLogLikelihood = logLikelihood;  // log P(observations | initial model).
	finalLogLikelihood = logLikelihood;

	//
	size_t i, k;
	double denominatorPi0 = 1.0 - double(K_);
	for (k = 0; k < K_; ++k)
		denominatorPi0 += (*pi_conj_)(k);
	dvector_type denominatorA0(K_, -double(K_));
	for (k = 0; k < K_; ++k)
		for (i = 0; i < K_; ++i)
			denominatorA0(k) += (*A_conj_)(k, i);

	double numeratorA, denominatorA;
	double delta;
	numIteration = 0;
	do
	{
		// M-step.
		for (k = 0; k < K_; ++k)
		{
			// reestimate frequency of state k in time n=0.
			pi_[k] = 0.001 + 0.999 * (gamma(0, k) + (*pi_conj_)(k) - 1.0) / denominatorPi0;

			// reestimate transition matrix.
			denominatorA = 0.0;
			for (n = 0; n < N - 1; ++n)
				denominatorA += gamma(n, k);

			for (i = 0; i < K_; ++i)
			{
				numeratorA = (*A_conj_)(k, i) - 1.0;
				for (n = 0; n < N - 1; ++n)
					numeratorA += xi[n](k, i);
				A_(k, i) = 0.001 + 0.999 * numeratorA / (denominatorA0(k) + denominatorA);
			}

			// reestimate observation(emission) distribution in each state.
			// run E-step & M-step as well.
			doEstimateObservationDensityParametersByMAPUsingConjugatePrior(N, (unsigned int)k, observations, gamma, denominatorA);
		}

		// E-step: evaluate gamma & xi.
		{
			// forward-backward algorithm.
			runForwardAlgorithm(N, observations, scale, alpha, logLikelihood);
			runBackwardAlgorithm(N, observations, scale, beta);

			computeGamma(N, alpha, beta, gamma);
			computeXi(N, observations, alpha, beta, xi);
		}

		// compute difference between log probability of two iterations.
#if 1
		delta = logLikelihood - finalLogLikelihood;
#else
		delta = std::fabs(logLikelihood - finalLogLikelihood);  // log P(observations | estimated model).
#endif

		finalLogLikelihood = logLikelihood;  // log P(observations | estimated model).
		++numIteration;
	} while (delta > terminationTolerance && numIteration <= maxIteration);  // if log probability does not change much, exit.

/*
	// compute gamma & xi.
	{
		// gamma can use the result from Baum-Welch algorithm.
		//dmatrix_type gamma2(boost::extents[N][K_]);
		//computeGamma(N, alpha, beta, gamma2);

		//
		std::vector<dmatrix_type> xi2;
		xi2.reserve(N - 1);
		for (n = 0; n < N - 1; ++n)
			xi2.push_back(dmatrix_type(K_, K_, 0.0));
		computeXi(N, observations, alpha, beta, xi2);
	}
*/

	return true;
}

bool CDHMM::trainByMAPUsingConjugatePrior(const std::vector<size_t> &Ns, const std::vector<dmatrix_type> &observationSequences, const double terminationTolerance, const size_t maxIteration, size_t &numIteration, std::vector<double> &initLogLikelihoods, std::vector<double> &finalLogLikelihoods)
{
	//	[ref] "Maximum a Posteriori Estimation for Multivariate Gaussian Mixture Observations of Markov Chains", J.-L. Gauvain adn C.-H. Lee, TSAP, 1994.

	if (!doDoHyperparametersOfConjugatePriorExist())
		throw std::runtime_error("Hyperparameters of the conjugate prior have to be assigned for MAP learning.");

	const size_t R = Ns.size();  // the number of observation sequences.
	size_t Nr, r, n;

	std::vector<dmatrix_type> alphas, betas, gammas;
	std::vector<std::vector<dmatrix_type> > xis;
	std::vector<dvector_type> scales;
	alphas.reserve(R);
	betas.reserve(R);
	gammas.reserve(R);
	xis.reserve(R);
	scales.reserve(R);
	for (r = 0; r < R; ++r)
	{
		Nr = Ns[r];
		alphas.push_back(dmatrix_type(Nr, K_, 0.0));
		betas.push_back(dmatrix_type(Nr, K_, 0.0));
		gammas.push_back(dmatrix_type(Nr, K_, 0.0));
		xis.push_back(std::vector<dmatrix_type>());
		xis[r].reserve(Nr - 1);
		for (n = 0; n < Nr - 1; ++n)
			xis[r].push_back(dmatrix_type(K_, K_, 0.0));
		scales.push_back(dvector_type(Nr, 0.0));
	}

	double logLikelihood;

	// E-step: evaluate gamma & xi.
	for (r = 0; r < R; ++r)
	{
		Nr = Ns[r];
		const dmatrix_type &observations = observationSequences[r];

		dmatrix_type &alphar = alphas[r];
		dmatrix_type &betar = betas[r];
		dmatrix_type &gammar = gammas[r];
		std::vector<dmatrix_type> &xir = xis[r];
		dvector_type &scaler = scales[r];

		// forward-backward algorithm.
		runForwardAlgorithm(Nr, observations, scaler, alphar, logLikelihood);
		runBackwardAlgorithm(Nr, observations, scaler, betar);

		computeGamma(Nr, alphar, betar, gammar);
		computeXi(Nr, observations, alphar, betar, xir);

		initLogLikelihoods[r] = logLikelihood;  // log P(observations | initial model).
		finalLogLikelihoods[r] = logLikelihood;
	}

	//
	size_t i, k;
	double denominatorPi0 = double(R) - double(K_);
	for (k = 0; k < K_; ++k)
		denominatorPi0 += (*pi_conj_)(k);
	dvector_type denominatorA0(K_, -double(K_));
	for (k = 0; k < K_; ++k)
		for (i = 0; i < K_; ++i)
			denominatorA0(k) += (*A_conj_)(k, i);

	double numeratorPi;
	double numeratorA, denominatorA;
	double delta;;
	bool continueToLoop;
	numIteration = 0;
	do
	{
		// M-step.
		for (k = 0; k < K_; ++k)
		{
			// reestimate frequency of state k in time n=0.
			numeratorPi = (*pi_conj_)(k) - 1.0;
			for (r = 0; r < R; ++r)
				numeratorPi += gammas[r](0, k);
			pi_[k] = 0.001 + 0.999 * numeratorPi / denominatorPi0;

			// reestimate transition matrix.
			denominatorA = 0.0;
			for (r = 0; r < R; ++r)
				for (n = 0; n < Ns[r] - 1; ++n)
					denominatorA += gammas[r](n, k);

			for (i = 0; i < K_; ++i)
			{
				numeratorA = (*A_conj_)(k, i) - 1.0;
				for (r = 0; r < R; ++r)
					for (n = 0; n < Ns[r] - 1; ++n)
						numeratorA += xis[r][n](k, i);
				A_(k, i) = 0.001 + 0.999 * numeratorA / (denominatorA0(k) + denominatorA);
			}

			// reestimate observation(emission) distribution in each state.
			// run E-step & M-step as well.
			doEstimateObservationDensityParametersByMAPUsingConjugatePrior(Ns, (unsigned int)k, observationSequences, gammas, R, denominatorA);
		}

		// E-step: evaluate gamma & xi.
		continueToLoop = false;
		for (r = 0; r < R; ++r)
		{
			Nr = Ns[r];
			const dmatrix_type &observations = observationSequences[r];

			dmatrix_type &alphar = alphas[r];
			dmatrix_type &betar = betas[r];
			dmatrix_type &gammar = gammas[r];
			std::vector<dmatrix_type> &xir = xis[r];
			dvector_type &scaler = scales[r];

			// forward-backward algorithm.
			runForwardAlgorithm(Nr, observations, scaler, alphar, logLikelihood);
			runBackwardAlgorithm(Nr, observations, scaler, betar);

			computeGamma(Nr, alphar, betar, gammar);
			computeXi(Nr, observations, alphar, betar, xir);

			// compute difference between log probability of two iterations.
#if 1
			delta = logLikelihood - finalLogLikelihoods[r];
#else
			delta = std::fabs(logLikelihood - finalLogLikelihoods[r]);
#endif
			if (delta > terminationTolerance && numIteration <= maxIteration)
				continueToLoop = true;

			finalLogLikelihoods[r] = logLikelihood;  // log P(observations | estimated model).
		}

		++numIteration;
	} while (continueToLoop);  // if log probability does not change much, exit.

/*
	// compute gamma & xi.
	{
		for (r = 0; r < R; ++r)
		{
			// gamma can use the result from Baum-Welch algorithm.
			//dmatrix_type gamma2(Ns[r], K_, 0.0);
			//computeGamma(Ns[r], alphas[r], betas[r], gamma2);

			//
			std::vector<std::vector<dmatrix_type> > xis2;
			xis2.reserve(R);
			for (r = 0; r < R; ++r)
			{
				Nr = Ns[r];
				xis2.push_back(std::vector<dmatrix_type>());
				xis2[r].reserve(Nr - 1);
				for (n = 0; n < Nr - 1; ++n)
					xis2[r].push_back(dmatrix_type(K_, K_, 0.0));
			}
			computeXi(Ns[r], observationSequences[r], alphas[r], betas[r], xis2[r]);
		}
	}
*/

	return true;
}

bool CDHMM::trainByMAPUsingEntropicPrior(const size_t N, const dmatrix_type &observations, const double z, const bool doesTrimParameter, const double terminationTolerance, const size_t maxIteration, size_t &numIteration, double &initLogLikelihood, double &finalLogLikelihood)
{
	// [ref] "Structure Learning in Conditional Probability Models via an Entropic Prior and Parameter Extinction", M. Brand, Neural Computation, 1999.
	// [ref] "Pattern discovery via entropy minimization", M. Brand, AISTATS, 1999.

	//if (!doDoHyperparametersOfEntropicPriorExist())
	//	throw std::runtime_error("Hyperparameters of the entropic prior have to be assigned for MAP learning.");

	dvector_type scale(N, 0.0);
	double logLikelihood;
	size_t n;

	dmatrix_type alpha(N, K_, 0.0), beta(N, K_, 0.0), gamma(N, K_, 0.0);
	std::vector<dmatrix_type> xi;
	xi.reserve(N - 1);
	for (n = 0; n < N - 1; ++n)
		xi.push_back(dmatrix_type(K_, K_, 0.0));

	// E-step: evaluate gamma & xi.
	{
		// forward-backward algorithm.
		runForwardAlgorithm(N, observations, scale, alpha, logLikelihood);
		runBackwardAlgorithm(N, observations, scale, beta);

		computeGamma(N, alpha, beta, gamma);
		computeXi(N, observations, alpha, beta, xi);
	}

	initLogLikelihood = logLikelihood;  // log P(observations | initial model).
	finalLogLikelihood = logLikelihood;

	//
	const double eps = 1e-50;

#if 0
	std::vector<double> omega(K_, 0.0), theta(K_, 0.0);
#else
	dvector_type expNumVisits1(K_, 0.0), expNumVisitsN(K_, 0.0);
	dmatrix_type expNumTrans(K_, K_, 0.0); //, expNumEmit(K_, D_, 0.0);
	std::vector<double> thetaTrans(K_, 0.0), thetaEmit(D_, 0.0);
	bool isNormalized;
	double grad;
#endif

	std::vector<bool> isTransitionsTrimmed, isObservationsTrimmed; //, isStatesTrimmed;
	if (doesTrimParameter && std::fabs(z - 1.0) <= eps)
	{
		isTransitionsTrimmed.resize(K_, false);
		isObservationsTrimmed.resize(K_, false);
		//isStatesTrimmed.resize(K_, false);
	}

	double denominatorA;
	double delta;
	double entropicMAPLogLikelihood = 0.0;
	double sumTheta, sumPi;
	size_t i, k;
	numIteration = 0;
	do
	{
		// M-step.
#if 0
		for (k = 0; k < K_; ++k)
		{
			// reestimate frequency of state k in time n=0.
			pi_[k] = 0.001 + 0.999 * gamma(0, k);

			// reestimate transition matrix.
			for (i = 0; i < K_; ++i)
			{
				omega[i] = 0.0;
				for (n = 0; n < N - 1; ++n)
					omega[i] += xi[n](k, i);
			}

			const bool retval = computeMAPEstimateOfMultinomialUsingEntropicPrior(omega, z, theta, entropicMAPLogLikelihood, terminationTolerance, maxIteration, true);
			assert(retval);

			// trim transition probabilities.
			if (doesTrimParameter && std::fabs(z - 1.0) <= eps)
			{
				throw std::runtime_error("not yet implemented");
			}

			for (i = 0; i < K_; ++i)
				A_(k, i) = theta[i];

			// reestimate observation(emission) distribution in each state.
			// run E-step & M-step as well.
			denominatorA = 0.0;
			for (n = 0; n < N - 1; ++n)
				denominatorA += gamma(n, k);

			doEstimateObservationDensityParametersByMAPUsingEntropicPrior(N, (unsigned int)k, observations, gamma, z, doesTrimParameter, terminationTolerance, maxIteration, denominatorA);
		}
#else
		{
			// compute expected sufficient statistics (ESS).
			expNumVisits1.clear();
			expNumVisitsN.clear();
			expNumTrans.clear();
			//expNumEmit.clear();
			doComputeExpectedSufficientStatistics(N, observations, gamma, xi, expNumVisits1, expNumVisitsN, expNumTrans/*, expNumEmit*/);
			sumPi = std::accumulate(expNumVisits1.begin(), expNumVisits1.end(), 0.0);
			assert(std::fabs(sumPi) >= eps);

			for (k = 0; k < K_; ++k)
			{
				// reestimate frequency of state k in time n=0.
#if 0
				pi_[k] = 0.001 + 0.999 * gamma(0, k);
#else
				pi_[k] = expNumVisits1[k] / sumPi;
#endif

				// reestimate transition matrix in each state.
				const bool retval = computeMAPEstimateOfMultinomialUsingEntropicPrior(boost::numeric::ublas::matrix_row<dmatrix_type>(expNumTrans, k), z, thetaTrans, entropicMAPLogLikelihood, terminationTolerance, maxIteration, true);
				assert(retval);

				// trim transition probabilities.
				//	only trim if we are in the min. entropy setting (z = 1).
				//	if z << 0, we would trim everything.
				if (doesTrimParameter && std::fabs(z - 1.0) <= eps)
				{
					isNormalized = false;
					if (!isTransitionsTrimmed[k])  // not yet trimmed.
					{
						for (i = 0; i < K_; ++i)
						{
							grad = std::fabs(thetaTrans[i]) < eps ? expNumTrans(k, i) : (expNumTrans(k, i) / thetaTrans[i]);
							if (thetaTrans[i] <= std::exp(-grad / z))
							{
								thetaTrans[i] = 0.0;
								isTransitionsTrimmed[k] = true;
								isNormalized = true;
							}
						}
					}

					if (isNormalized)
					{
						sumTheta = std::accumulate(thetaTrans.begin(), thetaTrans.end(), 0.0);
						assert(std::fabs(sumTheta) >= eps);
						for (i = 0; i < K_; ++i)
							A_(k, i) = thetaTrans[i] / sumTheta;
					}
					else
					{
						for (i = 0; i < K_; ++i)
							A_(k, i) = thetaTrans[i];
					}
				}
				else
				{
					for (i = 0; i < K_; ++i)
						A_(k, i) = thetaTrans[i];
				}

				// reestimate observation(emission) distribution in each state.
				// run E-step & M-step as well.
				denominatorA = 0.0;
				for (n = 0; n < N - 1; ++n)
					denominatorA += gamma(n, k);

				doEstimateObservationDensityParametersByMAPUsingEntropicPrior(N, (unsigned int)k, observations, gamma, z, doesTrimParameter, terminationTolerance, maxIteration, denominatorA);
			}
		}
#endif

		// E-step: evaluate gamma & xi.
		{
			// forward-backward algorithm.
			runForwardAlgorithm(N, observations, scale, alpha, logLikelihood);
			runBackwardAlgorithm(N, observations, scale, beta);

			computeGamma(N, alpha, beta, gamma);
			computeXi(N, observations, alpha, beta, xi);
		}

		// compute difference between log probability of two iterations.
#if 1
		delta = logLikelihood - finalLogLikelihood;
#else
		delta = std::fabs(logLikelihood - finalLogLikelihood);  // log P(observations | estimated model).
#endif

		finalLogLikelihood = logLikelihood;  // log P(observations | estimated model).
		++numIteration;
	} while (delta > terminationTolerance && numIteration <= maxIteration);  // if log probability does not change much, exit.

/*
	// compute gamma & xi.
	{
		// gamma can use the result from Baum-Welch algorithm.
		//dmatrix_type gamma2(boost::extents[N][K_]);
		//computeGamma(N, alpha, beta, gamma2);

		//
		std::vector<dmatrix_type> xi2;
		xi2.reserve(N - 1);
		for (n = 0; n < N - 1; ++n)
			xi2.push_back(dmatrix_type(K_, K_, 0.0));
		computeXi(N, observations, alpha, beta, xi2);
	}
*/

	return true;
}

bool CDHMM::trainByMAPUsingEntropicPrior(const std::vector<size_t> &Ns, const std::vector<dmatrix_type> &observationSequences, const double z, const bool doesTrimParameter, const double terminationTolerance, const size_t maxIteration, size_t &numIteration, std::vector<double> &initLogLikelihoods, std::vector<double> &finalLogLikelihoods)
{
	// [ref] "Structure Learning in Conditional Probability Models via an Entropic Prior and Parameter Extinction", M. Brand, Neural Computation, 1999.
	// [ref] "Pattern discovery via entropy minimization", M. Brand, AISTATS, 1999.

	//if (!doDoHyperparametersOfEntropicPriorExist())
	//	throw std::runtime_error("Hyperparameters of the entropic prior have to be assigned for MAP learning.");

	const size_t R = Ns.size();  // the number of observation sequences.
	size_t Nr, r, n;

	std::vector<dmatrix_type> alphas, betas, gammas;
	std::vector<std::vector<dmatrix_type> > xis;
	std::vector<dvector_type> scales;
	alphas.reserve(R);
	betas.reserve(R);
	gammas.reserve(R);
	xis.reserve(R);
	scales.reserve(R);
	for (r = 0; r < R; ++r)
	{
		Nr = Ns[r];
		alphas.push_back(dmatrix_type(Nr, K_, 0.0));
		betas.push_back(dmatrix_type(Nr, K_, 0.0));
		gammas.push_back(dmatrix_type(Nr, K_, 0.0));
		xis.push_back(std::vector<dmatrix_type>());
		xis[r].reserve(Nr - 1);
		for (n = 0; n < Nr - 1; ++n)
			xis[r].push_back(dmatrix_type(K_, K_, 0.0));
		scales.push_back(dvector_type(Nr, 0.0));
	}

	double logLikelihood;

	// E-step: evaluate gamma & xi.
	for (r = 0; r < R; ++r)
	{
		Nr = Ns[r];
		const dmatrix_type &observations = observationSequences[r];

		dmatrix_type &alphar = alphas[r];
		dmatrix_type &betar = betas[r];
		dmatrix_type &gammar = gammas[r];
		std::vector<dmatrix_type> &xir = xis[r];
		dvector_type &scaler = scales[r];

		// forward-backward algorithm.
		runForwardAlgorithm(Nr, observations, scaler, alphar, logLikelihood);
		runBackwardAlgorithm(Nr, observations, scaler, betar);

		computeGamma(Nr, alphar, betar, gammar);
		computeXi(Nr, observations, alphar, betar, xir);

		initLogLikelihoods[r] = logLikelihood;  // log P(observations | initial model).
		finalLogLikelihoods[r] = logLikelihood;
	}

	//
	const double eps = 1e-50;

#if 0
	std::vector<double> omega(K_, 0.0), theta(K_, 0.0);
#else
	dvector_type expNumVisits1(K_, 0.0), expNumVisitsN(K_, 0.0);
	dmatrix_type expNumTrans(K_, K_, 0.0); //, expNumEmit(K_, D_, 0.0);
	std::vector<double> thetaTrans(K_, 0.0), thetaEmit(D_, 0.0);
	bool isNormalized;
	double grad;
#endif

	std::vector<bool> isTransitionsTrimmed, isObservationsTrimmed; //, isStatesTrimmed;
	if (doesTrimParameter && std::fabs(z - 1.0) <= eps)
	{
		isTransitionsTrimmed.resize(K_, false);
		isObservationsTrimmed.resize(K_, false);
		//isStatesTrimmed.resize(K_, false);
	}

	//double numeratorPi;
	double denominatorA;
	double delta;;
	bool continueToLoop;
	std::vector<double> omega(K_, 0.0), theta(K_, 0.0);
	double entropicMAPLogLikelihood = 0.0;
	double sumTheta, sumPi;
	size_t i, k;
	numIteration = 0;
	do
	{
		// M-step.
#if 0
		for (k = 0; k < K_; ++k)
		{
			// reestimate frequency of state k in time n=0.
			numeratorPi = 0.0;
			for (r = 0; r < R; ++r)
				numeratorPi += gammas[r](0, k);
			pi_[k] = 0.001 + 0.999 * numeratorPi / (double)R;

			// reestimate transition matrix.
			for (i = 0; i < K_; ++i)
			{
				omega[i] = 0.0;
				for (r = 0; r < R; ++r)
					for (n = 0; n < Ns[r] - 1; ++n)
						omega[i] += xis[r][n](k, i);
			}

			const bool retval = computeMAPEstimateOfMultinomialUsingEntropicPrior(omega, z, theta, entropicMAPLogLikelihood, terminationTolerance, maxIteration, true);
			assert(retval);

			// trim transition probabilities.
			if (doesTrimParameter && std::fabs(z - 1.0) <= eps)
			{
				throw std::runtime_error("not yet implemented");
			}

			for (i = 0; i < K_; ++i)
				A_(k, i) = theta[i];

			// reestimate observation(emission) distribution in each state.
			// run E-step & M-step as well.
			denominatorA = 0.0;
			for (r = 0; r < R; ++r)
				for (n = 0; n < Ns[r] - 1; ++n)
					denominatorA += gammas[r](n, k);

			doEstimateObservationDensityParametersByMAPUsingEntropicPrior(Ns, (unsigned int)k, observationSequences, gammas, z, doesTrimParameter, terminationTolerance, maxIteration, R, denominatorA);
		}
#else
		{
			// compute expected sufficient statistics (ESS).
			expNumVisits1.clear();
			expNumVisitsN.clear();
			expNumTrans.clear();
			//expNumEmit.clear();
			doComputeExpectedSufficientStatistics(Ns, observationSequences, gammas, xis, expNumVisits1, expNumVisitsN, expNumTrans/*, expNumEmit*/);
			sumPi = std::accumulate(expNumVisits1.begin(), expNumVisits1.end(), 0.0);
			assert(std::fabs(sumPi) >= eps);

			for (k = 0; k < K_; ++k)
			{
				// reestimate frequency of state k in time n=0.
#if 0
				numeratorPi = 0.0;
				for (r = 0; r < R; ++r)
					numeratorPi += gammas[r](0, k);
				pi_[k] = 0.001 + 0.999 * numeratorPi / (double)R;
#else
				pi_[k] = expNumVisits1[k] / sumPi;
#endif

				// reestimate transition matrix in each state.
				const bool retval = computeMAPEstimateOfMultinomialUsingEntropicPrior(boost::numeric::ublas::matrix_row<dmatrix_type>(expNumTrans, k), z, thetaTrans, entropicMAPLogLikelihood, terminationTolerance, maxIteration, true);
				assert(retval);

				// trim transition probabilities.
				//	only trim if we are in the min. entropy setting (z = 1).
				//	if z << 0, we would trim everything.
				if (doesTrimParameter && std::fabs(z - 1.0) <= eps)
				{
					isNormalized = false;
					if (!isTransitionsTrimmed[k])  // not yet trimmed.
					{
						for (i = 0; i < K_; ++i)
						{
							grad = std::fabs(thetaTrans[i]) < eps ? expNumTrans(k, i) : (expNumTrans(k, i) / thetaTrans[i]);
							if (thetaTrans[i] <= std::exp(-grad / z))
							{
								thetaTrans[i] = 0.0;
								isTransitionsTrimmed[k] = true;
								isNormalized = true;
							}
						}
					}

					if (isNormalized)
					{
						sumTheta = std::accumulate(thetaTrans.begin(), thetaTrans.end(), 0.0);
						assert(std::fabs(sumTheta) >= eps);
						for (i = 0; i < K_; ++i)
							A_(k, i) = thetaTrans[i] / sumTheta;
					}
					else
					{
						for (i = 0; i < K_; ++i)
							A_(k, i) = thetaTrans[i];
					}
				}
				else
				{
					for (i = 0; i < K_; ++i)
						A_(k, i) = thetaTrans[i];
				}

				// reestimate observation(emission) distribution in each state.
				// run E-step & M-step as well.
				denominatorA = 0.0;
				for (r = 0; r < R; ++r)
					for (n = 0; n < Ns[r] - 1; ++n)
						denominatorA += gammas[r](n, k);

				doEstimateObservationDensityParametersByMAPUsingEntropicPrior(Ns, (unsigned int)k, observationSequences, gammas, z, doesTrimParameter, terminationTolerance, maxIteration, R, denominatorA);
			}
		}
#endif

		// E-step: evaluate gamma & xi.
		continueToLoop = false;
		for (r = 0; r < R; ++r)
		{
			Nr = Ns[r];
			const dmatrix_type &observations = observationSequences[r];

			dmatrix_type &alphar = alphas[r];
			dmatrix_type &betar = betas[r];
			dmatrix_type &gammar = gammas[r];
			std::vector<dmatrix_type> &xir = xis[r];
			dvector_type &scaler = scales[r];

			// forward-backward algorithm.
			runForwardAlgorithm(Nr, observations, scaler, alphar, logLikelihood);
			runBackwardAlgorithm(Nr, observations, scaler, betar);

			computeGamma(Nr, alphar, betar, gammar);
			computeXi(Nr, observations, alphar, betar, xir);

			// compute difference between log probability of two iterations.
#if 1
			delta = logLikelihood - finalLogLikelihoods[r];
#else
			delta = std::fabs(logLikelihood - finalLogLikelihoods[r]);
#endif
			if (delta > terminationTolerance && numIteration <= maxIteration)
				continueToLoop = true;

			finalLogLikelihoods[r] = logLikelihood;  // log P(observations | estimated model).
		}

		++numIteration;
	} while (continueToLoop);  // if log probability does not change much, exit.

/*
	// compute gamma & xi.
	{
		for (r = 0; r < R; ++r)
		{
			// gamma can use the result from Baum-Welch algorithm.
			//dmatrix_type gamma2(Ns[r], K_, 0.0);
			//computeGamma(Ns[r], alphas[r], betas[r], gamma2);

			//
			std::vector<std::vector<dmatrix_type> > xis2;
			xis2.reserve(R);
			for (r = 0; r < R; ++r)
			{
				Nr = Ns[r];
				xis2.push_back(std::vector<dmatrix_type>());
				xis2[r].reserve(Nr - 1);
				for (n = 0; n < Nr - 1; ++n)
					xis2[r].push_back(dmatrix_type(K_, K_, 0.0));
			}
			computeXi(Ns[r], observationSequences[r], alphas[r], betas[r], xis2[r]);
		}
	}
*/

	return true;
}

void CDHMM::computeXi(const size_t N, const dmatrix_type &observations, const dmatrix_type &alpha, const dmatrix_type &beta, std::vector<dmatrix_type> &xi) const
{
	size_t i, k;
	double sum;
	for (size_t n = 0; n < N - 1; ++n)
	{
		sum = 0.0;
		for (k = 0; k < K_; ++k)
			for (i = 0; i < K_; ++i)
			{
				//xi[n](k, i) = alpha(n, k) * beta(n+1, i) * A_(k, i) * B_(i, observations[n+1]);
				//xi[n](k, i) = alpha(n, k) * beta(n+1, i) * A_(k, i) * doEvaluateEmissionProbability(i, boost::numeric::ublas::matrix_row<const dmatrix_type>(observations, n+1));
				xi[n](k, i) = alpha(n, k) * beta(n+1, i) * A_(k, i) * doEvaluateEmissionProbability(i, n+1, observations);
				sum += xi[n](k, i);
			}

		for (k = 0; k < K_; ++k)
			for (i = 0; i < K_; ++i)
				xi[n](k, i) /= sum;
	}
}

void CDHMM::doComputeObservationLikelihood(const size_t N, const dmatrix_type &observations, dmatrix_type &obsLikelihood) const
{
	// PRECONDITIONS [] >>
	//	-. obsLikelihood is allocated and initialized before this function is called.

	// [ref] mk_dhmm_obs_lik in http://www.merl.com/people/brand/ or http://mcgill-android-parking.googlecode.com/svn/trunk/MatLab_v1.3/.

	// function B = mk_dhmm_obs_lik(data, obsmat, obsmat1)
	//
	// MK_DHMM_OBS_LIK  Make the observation likelihood vector for a discrete HMM.
	//
	// Inputs:
	// data(n) = x(n) = observation at time n
	// obsmat(i,o) = Pr(x(n)=o | z(n)=i)
	// obsmat1(i,o) = Pr(x(1)=o | z(1)=i). Defaults to obsmat if omitted.
	//
	// Output:
	// B(i,n) = Pr(x(n) | z(t)=i)

	size_t n, k;
	for (n = 0; n < N; ++n)
		for (k = 0; k < K_; ++k)
			//obsLikelihood(k, n) = doEvaluateEmissionProbability(k, boost::numeric::ublas::matrix_row<const dmatrix_type>(observations, n));
			obsLikelihood(k, n) = doEvaluateEmissionProbability(k, n, observations);
}

void CDHMM::doComputeExpectedSufficientStatistics(const size_t N, const dmatrix_type &observations, const dmatrix_type &gamma, const std::vector<dmatrix_type> &xi, dvector_type &expNumVisits1, dvector_type &expNumVisitsN, dmatrix_type &expNumTrans/*, dmatrix_type &expNumEmit*/) const
{
	// PRECONDITIONS [] >>
	//	-. expNumVisits1, expNumVisitsN, expNumTrans, and expNumEmit are allocated and initialized before this function is called.

	// [ref] compute_ess_dhmm in http://www.merl.com/people/brand/ or http://mcgill-android-parking.googlecode.com/svn/trunk/MatLab_v1.3/.

	// function [loglik, exp_num_trans, exp_num_visits1, exp_num_emit, exp_num_visitsN] = ...
	//	compute_ess_dhmm(startprob, transmat, obsmat, data, dirichlet)
	//
	// Compute the Expected Sufficient Statistics for a discrete Hidden Markov Model.
	//
	// Outputs:
	// exp_num_trans(i,j) = sum_{n=2}^N Pr(z(n-1)=i, z(n)=j | Obs)
	// exp_num_visits1(i) = Pr(z(1)=i | Obs)
	// exp_num_visitsN(i) = Pr(z(N)=i | Obs) 
	// exp_num_emit(i,o) = sum_{n=1}^N Pr(z(n)=i, x(n)=o | Obs)
	// where Obs = O_1 .. O_N for observation sequence.

	expNumVisits1 += boost::numeric::ublas::matrix_row<dmatrix_type>((dmatrix_type &)gamma, 0);
	expNumVisitsN += boost::numeric::ublas::matrix_row<dmatrix_type>((dmatrix_type &)gamma, N - 1);

	size_t n;
	for (n = 0; n < N; ++n)
	{
		if (n < N - 1) expNumTrans += xi[n];
/*
		for (k = 0; k < K_; ++k)
			expNumEmit(k, observations[n]) += boost::numeric::ublas::matrix_row<dmatrix_type>((dmatrix_type &)gamma, n);
*/
	}
}

void CDHMM::doComputeExpectedSufficientStatistics(const std::vector<size_t> &Ns, const std::vector<dmatrix_type> &observationSequences, const std::vector<dmatrix_type> &gammas, const std::vector<std::vector<dmatrix_type> > &xis, dvector_type &expNumVisits1, dvector_type &expNumVisitsN, dmatrix_type &expNumTrans/*, dmatrix_type &expNumEmit*/) const
{
	// PRECONDITIONS [] >>
	//	-. expNumVisits1, expNumVisitsN, expNumTrans, and expNumEmit are allocated and initialized before this function is called.

	// [ref] compute_ess_dhmm in http://www.merl.com/people/brand/ or http://mcgill-android-parking.googlecode.com/svn/trunk/MatLab_v1.3/.

	// function [loglik, exp_num_trans, exp_num_visits1, exp_num_emit, exp_num_visitsN] = ...
	//	compute_ess_dhmm(startprob, transmat, obsmat, data, dirichlet)
	//
	// Compute the Expected Sufficient Statistics for a discrete Hidden Markov Model.
	//
	// Outputs:
	// exp_num_trans(i,j) = sum_r sum_{n=2}^N Pr(z(n-1)=i, z(n)=j | Obs(r))
	// exp_num_visits1(i) = sum_r Pr(z(1)=i | Obs(r))
	// exp_num_visitsN(i) = sum_r Pr(z(N)=i | Obs(r)) 
	// exp_num_emit(i,o) = sum_r sum_{n=1}^N Pr(z(n)=i, x(n)=o | Obs(r))
	// where Obs(r) = O_1 .. O_N for sequence r.

	const size_t R = Ns.size();  // number of observations sequences.
	for (size_t r = 0; r < R; ++r)
		doComputeExpectedSufficientStatistics(Ns[r], observationSequences[r], gammas[r], xis[r], expNumVisits1, expNumVisitsN, expNumTrans/*, expNumEmit*/);
}

double CDHMM::doEvaluateEmissionProbability(const unsigned int state, const size_t n, const dmatrix_type &observations) const
{
	return doEvaluateEmissionProbability(state, boost::numeric::ublas::matrix_row<const dmatrix_type>(observations, n));
}

void CDHMM::generateSample(const size_t N, dmatrix_type &observations, uivector_type &states, const unsigned int seed /*= (unsigned int)-1*/) const
{
	// PRECONDITIONS [] >>
	//	-. std::srand() has to be called before this function is called.

	doInitializeRandomSampleGeneration(seed);

	states[0] = generateInitialState();
	doGenerateObservationsSymbol(states[0], 0, observations);

	for (size_t n = 1; n < N; ++n)
	{
		states[n] = generateNextState(states[n-1]);
		doGenerateObservationsSymbol(states[n], n, observations);
	}

	doFinalizeRandomSampleGeneration();
}

/*static*/ bool CDHMM::readSequence(std::istream &stream, size_t &N, size_t &D, dmatrix_type &observations)
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
	for (size_t n = 0; n < N; ++n)
		for (size_t i = 0; i < D; ++i)
			stream >> observations(n, i);

	return true;
}

/*static*/ bool CDHMM::writeSequence(std::ostream &stream, const dmatrix_type &observations)
{
	const size_t N = observations.size1();
	const size_t D = observations.size2();

	stream << "N= " << N << std::endl;
	stream << "D= " << D << std::endl;
	for (size_t n = 0; n < N; ++n)
	{
		for (size_t i = 0; i < D; ++i)
			stream << observations(n, i) << ' ';
		stream << std::endl;
	}

	return true;
}

}  // namespace swl
