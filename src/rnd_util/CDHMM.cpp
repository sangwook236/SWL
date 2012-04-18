#include "swl/Config.h"
#include "swl/rnd_util/CDHMM.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

CDHMM::CDHMM(const size_t K, const size_t D)
: base_type(K, D)
{
}

CDHMM::CDHMM(const size_t K, const size_t D, const std::vector<double> &pi, const boost::multi_array<double, 2> &A)
: base_type(K, D, pi, A)
{
}

CDHMM::~CDHMM()
{
}

void CDHMM::runForwardAlgorithm(const size_t N, const boost::multi_array<double, 2> &observations, boost::multi_array<double, 2> &alpha, double &probability) const
{
	size_t i, k;  // state indices

	// 1. Initialization
	for (k = 0; k < K_; ++k)
		//alpha[0][k] = pi_[k] * B_[k][observations[0]];
		alpha[0][k] = pi_[k] * doEvaluateEmissionProbability(k, observations[boost::indices[0][boost::multi_array<double, 2>::index_range()]]);

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
				sum += alpha[n_1][i] * A_[i][k];

			//alpha[n][k] = sum * B_[k][observations[n]];
			alpha[n][k] = sum * doEvaluateEmissionProbability(k, observations[boost::indices[n][boost::multi_array<double, 2>::index_range()]]);
		}
	}

	// 3. Termination
	probability = 0.0;
	n_1 = N - 1;
	for (k = 0; k < K_; ++k)
		probability += alpha[n_1][k];
}

void CDHMM::runForwardAlgorithm(const size_t N, const boost::multi_array<double, 2> &observations, std::vector<double> &scale, boost::multi_array<double, 2> &alpha, double &probability) const
{
	size_t i, k;  // state indices

	// 1. Initialization
	scale[0] = 0.0;
	for (k = 0; k < K_; ++k)
	{
		//alpha[0][k] = pi_[k] * B_[k][observations[0]];
		alpha[0][k] = pi_[k] * doEvaluateEmissionProbability(k, observations[boost::indices[0][boost::multi_array<double, 2>::index_range()]]);
		scale[0] += alpha[0][k];
	}
	for (k = 0; k < K_; ++k)
		alpha[0][k] /= scale[0];

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
				sum += alpha[n_1][i] * A_[i][k];

			//alpha[n][k] = sum * B_[k][observations[n]];
			alpha[n][k] = sum * doEvaluateEmissionProbability(k, observations[boost::indices[n][boost::multi_array<double, 2>::index_range()]]);
			scale[n] += alpha[n][k];
		}
		for (k = 0; k < K_; ++k)
			alpha[n][k] /= scale[n];
	}

	// 3. Termination
	probability = 0.0;
	for (n = 0; n < N; ++n)
		probability += std::log(scale[n]);
}

void CDHMM::runBackwardAlgorithm(const size_t N, const boost::multi_array<double, 2> &observations, boost::multi_array<double, 2> &beta, double &probability) const
{
	size_t i, k;  // state indices
	size_t n_1;

	// 1. Initialization
	n_1 = N - 1;
	for (k = 0; k < K_; ++k)
		beta[n_1][k] = 1.0;

	// 2. Induction
	double sum;
	for (size_t n = N - 1; n > 0; --n)
	{
		n_1 = n - 1;
		for (k = 0; k < K_; ++k)
		{
			sum = 0.0;
			for (i = 0; i < K_; ++i)
				//sum += A_[k][i] * B_[i][observations[n]] * beta[n][i];
				sum += A_[k][i] * doEvaluateEmissionProbability(i, observations[boost::indices[n][boost::multi_array<double, 2>::index_range()]]) * beta[n][i];
			beta[n_1][k] = sum;
		}
	}

	// 3. Termination
	probability = 0.0;
	for (k = 0; k < K_; ++k)
		probability += beta[0][k];
}

void CDHMM::runBackwardAlgorithm(const size_t N, const boost::multi_array<double, 2> &observations, const std::vector<double> &scale, boost::multi_array<double, 2> &beta, double &probability) const
{
	size_t i, k;  // state indices
	size_t n_1;

	// 1. Initialization
	n_1 = N - 1;
	for (k = 0; k < K_; ++k)
		beta[n_1][k] = 1.0 / scale[n_1];

	// 2. Induction
	double sum;
	for (size_t n = N - 1; n > 0; --n)
	{
		n_1 = n - 1;
		for (k = 0; k < K_; ++k)
		{
			sum = 0.0;
			for (i = 0; i < K_; ++i)
				//sum += A_[k][i] * B_[i][observations[n]] * beta[n][i];
				sum += A_[k][i] * doEvaluateEmissionProbability(i, observations[boost::indices[n][boost::multi_array<double, 2>::index_range()]]) * beta[n][i];
			beta[n_1][k] = sum / scale[n];
		}
	}
}

void CDHMM::runViterbiAlgorithm(const size_t N, const boost::multi_array<double, 2> &observations, boost::multi_array<double, 2> &delta, boost::multi_array<unsigned int, 2> &psi, std::vector<unsigned int> &states, double &probability, const bool useLog /*= true*/) const
{
	if (useLog) runViterbiAlgorithmUsingLog(N, observations, delta, psi, states, probability);
	else runViterbiAlgorithmNotUsigLog(N, observations, delta, psi, states, probability);
}

void CDHMM::runViterbiAlgorithmNotUsigLog(const size_t N, const boost::multi_array<double, 2> &observations, boost::multi_array<double, 2> &delta, boost::multi_array<unsigned int, 2> &psi, std::vector<unsigned int> &states, double &probability) const
{
	size_t i, k;  // state indices

	// 1. Initialization
	for (k = 0; k < K_; ++k)
	{
		//delta[0][k] = pi_[k] * B_[k][observations[0]];
		delta[0][k] = pi_[k] * doEvaluateEmissionProbability(k, observations[boost::indices[0][boost::multi_array<double, 2>::index_range()]]);
		psi[0][k] = 0u;
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
				val = delta[n_1][i] * A_[i][k];
				if (val > maxval)
				{
					maxval = val;
					maxvalind = i;
				}
			}

			//delta[n][k] = maxval * B_[k][observations[n]];
			delta[n][k] = maxval * doEvaluateEmissionProbability(k, observations[boost::indices[n][boost::multi_array<double, 2>::index_range()]]);
			psi[n][k] = (unsigned int)maxvalind;
		}
	}

	// 3. Termination
	probability = 0.0;
	n_1 = N - 1;
	states[n_1] = 0u;
	for (k = 0; k < K_; ++k)
	{
		if (delta[n_1][k] > probability)
		{
			probability = delta[n_1][k];
			states[n_1] = (unsigned int)k;
		}
	}

	// 4. Path (state sequence) backtracking
	for (n = N - 1; n > 0; --n)
		states[n-1] = psi[n][states[n]];
}

void CDHMM::runViterbiAlgorithmUsingLog(const size_t N, const boost::multi_array<double, 2> &observations, boost::multi_array<double, 2> &delta, boost::multi_array<unsigned int, 2> &psi, std::vector<unsigned int> &states, double &probability) const
{
	size_t i, k;  // state indices
	size_t n;

	// 0. Preprocessing
	std::vector<double> logPi(pi_);
	boost::multi_array<double, 2> logA(A_);
	boost::multi_array<double, 2> logO(boost::extents[K_][N]);
	for (k = 0; k < K_; ++k)
	{
		logPi[k] = std::log(pi_[k]);

		for (i = 0; i < K_; ++i)
			logA[k][i] = std::log(A_[k][i]);

		for (n = 0; n < N; ++n)
			//logO[k][n] = std::log(B_[k][observations[n]]);
			logO[k][n] = std::log(doEvaluateEmissionProbability(k, observations[boost::indices[n][boost::multi_array<double, 2>::index_range()]]));
	}

	// 1. Initialization
	for (k = 0; k < K_; ++k)
	{
		delta[0][k] = logPi[k] + logO[k][0];
		psi[0][k] = 0u;
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
				val = delta[n_1][i] + logA[i][k];
				if (val > maxval)
				{
					maxval = val;
					maxvalind = i;
				}
			}

			delta[n][k] = maxval + logO[k][n];
			psi[n][k] = (unsigned int)maxvalind;
		}
	}

	// 3. Termination
	probability = -std::numeric_limits<double>::max();
	n_1 = N - 1;
	states[n_1] = 0u;
	for (k = 0; k < K_; ++k)
	{
		if (delta[n_1][k] > probability)
		{
			probability = delta[n_1][k];
			states[n_1] = (unsigned int)k;
		}
	}

	// 4. Path (state sequence) backtracking
	for (n = N - 1; n > 0; --n)
		states[n-1] = psi[n][states[n]];
}

bool CDHMM::estimateParameters(const size_t N, const boost::multi_array<double, 2> &observations, const double terminationTolerance, boost::multi_array<double, 2> &alpha, boost::multi_array<double, 2> &beta, boost::multi_array<double, 2> &gamma, size_t &numIteration, double &initLogProbability, double &finalLogProbability)
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
			doEstimateObservationDensityParametersInMStep(N, observations, gamma, denominatorA, k);
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

bool CDHMM::estimateParameters(const std::vector<size_t> &Ns, const std::vector<boost::multi_array<double, 2> > &observationSequences, const double terminationTolerance, size_t &numIteration,std::vector<double> &initLogProbabilities, std::vector<double> &finalLogProbabilities)
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
			doEstimateObservationDensityParametersInMStep(Ns, observationSequences, gammas, R, denominatorA, k);
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

void CDHMM::computeXi(const size_t N, const boost::multi_array<double, 2> &observations, const boost::multi_array<double, 2> &alpha, const boost::multi_array<double, 2> &beta, boost::multi_array<double, 3> &xi) const
{
	size_t i, k;
	double sum;
	for (size_t n = 0; n < N - 1; ++n)
	{
		sum = 0.0;
		for (k = 0; k < K_; ++k)
			for (i = 0; i < K_; ++i)
			{
				//xi[n][k][i] = alpha[n][k] * beta[n+1][i] * (A_[k][i]) * (B_[i][observations[n+1]]);
				xi[n][k][i] = alpha[n][k] * beta[n+1][i] * (A_[k][i]) * doEvaluateEmissionProbability(i, observations[boost::indices[n+1][boost::multi_array<double, 2>::index_range()]]);
				sum += xi[n][k][i];
			}

		for (k = 0; k < K_; ++k)
			for (i = 0; i < K_; ++i)
				xi[n][k][i] /= sum;
	}
}

void CDHMM::generateSample(const size_t N, boost::multi_array<double, 2> &observations, std::vector<unsigned int> &states, const unsigned int seed /*= (unsigned int)-1*/) const
{
	// PRECONDITIONS [] >>
	//	-. std::srand() had to be called before this function is called.

	states[0] = generateInitialState();
	doGenerateObservationsSymbol(states[0], observations[boost::indices[0][boost::multi_array<double, 2>::index_range()]], seed);

	for (size_t n = 1; n < N; ++n)
	{
		states[n] = generateNextState(states[n-1]);
		doGenerateObservationsSymbol(states[n], observations[boost::indices[n][boost::multi_array<double, 2>::index_range()]], (unsigned int)-1);
	}
}

/*static*/ bool CDHMM::readSequence(std::istream &stream, size_t &N, size_t &D, boost::multi_array<double, 2> &observations)
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

	observations.resize(boost::extents[N][D]);
	for (size_t n = 0; n < N; ++n)
		for (size_t i = 0; i < D; ++i)
			stream >> observations[n][i];

	return true;
}

/*static*/ bool CDHMM::writeSequence(std::ostream &stream, const boost::multi_array<double, 2> &observations)
{
	const boost::multi_array<double, 2>::size_type *shape = observations.shape();
	const size_t N = shape[0];
	const size_t D = shape[1];

	stream << "N= " << N << std::endl;
	stream << "D= " << D << std::endl;
	for (size_t n = 0; n < N; ++n)
	{
		for (size_t i = 0; i < D; ++i)
			stream << observations[n][i] << ' ';
		stream << std::endl;
	}

	return true;
}

}  // namespace swl
