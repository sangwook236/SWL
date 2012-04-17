#include "swl/Config.h"
#include "swl/rnd_util/DDHMM.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

DDHMM::DDHMM(const size_t K, const size_t D)
: base_type(K, D)
{
}

DDHMM::DDHMM(const size_t K, const size_t D, const std::vector<double> &pi, const boost::multi_array<double, 2> &A)
: base_type(K, D, pi, A)
{
}

DDHMM::~DDHMM()
{
}

void DDHMM::runForwardAlgorithm(const size_t N, const std::vector<unsigned int> &observations, boost::multi_array<double, 2> &alpha, double &probability) const
{
	size_t i, k;  // state indices

	// 1. Initialization
	for (k = 0; k < K_; ++k)
		//alpha[0][k] = pi_[k] * B_[k][observations[0]];
		alpha[0][k] = pi_[k] * evaluateEmissionProbability(k, observations[0]);

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
			alpha[n][k] = sum * evaluateEmissionProbability(k, observations[n]);
		}
	}

	// 3. Termination
	probability = 0.0;
	n_1 = N - 1;
	for (k = 0; k < K_; ++k)
		probability += alpha[n_1][k];
}

void DDHMM::runForwardAlgorithm(const size_t N, const std::vector<unsigned int> &observations, std::vector<double> &scale, boost::multi_array<double, 2> &alpha, double &probability) const
{
	size_t i, k;  // state indices

	// 1. Initialization
	scale[0] = 0.0;
	for (k = 0; k < K_; ++k)
	{
		//alpha[0][k] = pi_[k] * B_[k][observations[0]];
		alpha[0][k] = pi_[k] * evaluateEmissionProbability(k, observations[0]);
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
			alpha[n][k] = sum * evaluateEmissionProbability(k, observations[n]);
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

void DDHMM::runBackwardAlgorithm(const size_t N, const std::vector<unsigned int> &observations, boost::multi_array<double, 2> &beta, double &probability) const
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
				sum += A_[k][i] * evaluateEmissionProbability(i, observations[n]) * beta[n][i];
			beta[n_1][k] = sum;
		}
	}

	// 3. Termination
	probability = 0.0;
	for (k = 0; k < K_; ++k)
		probability += beta[0][k];
}

void DDHMM::runBackwardAlgorithm(const size_t N, const std::vector<unsigned int> &observations, const std::vector<double> &scale, boost::multi_array<double, 2> &beta, double &probability) const
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
				sum += A_[k][i] * evaluateEmissionProbability(i, observations[n]) * beta[n][i];
			beta[n_1][k] = sum / scale[n_1];
		}
	}
}

void DDHMM::runViterbiAlgorithm(const size_t N, const std::vector<unsigned int> &observations, boost::multi_array<double, 2> &delta, boost::multi_array<unsigned int, 2> &psi, std::vector<unsigned int> &states, double &probability, const bool useLog /*= true*/) const
{
	if (useLog) runViterbiAlgorithmUsingLog(N, observations, delta, psi, states, probability);
	else runViterbiAlgorithmNotUsigLog(N, observations, delta, psi, states, probability);
}

void DDHMM::runViterbiAlgorithmNotUsigLog(const size_t N, const std::vector<unsigned int> &observations, boost::multi_array<double, 2> &delta, boost::multi_array<unsigned int, 2> &psi, std::vector<unsigned int> &states, double &probability) const
{
	size_t i, k;  // state indices

	// 1. Initialization
	for (k = 0; k < K_; ++k)
	{
		//delta[0][k] = pi_[k] * B_[k][observations[0]];
		delta[0][k] = pi_[k] * evaluateEmissionProbability(k, observations[0]);
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
			delta[n][k] = maxval * evaluateEmissionProbability(k, observations[n]);
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

void DDHMM::runViterbiAlgorithmUsingLog(const size_t N, const std::vector<unsigned int> &observations, boost::multi_array<double, 2> &delta, boost::multi_array<unsigned int, 2> &psi, std::vector<unsigned int> &states, double &probability) const
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
			logO[k][n] = std::log(evaluateEmissionProbability(k, observations[n]));
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

void DDHMM::computeXi(const size_t N, const std::vector<unsigned int> &observations, const boost::multi_array<double, 2> &alpha, const boost::multi_array<double, 2> &beta, boost::multi_array<double, 3> &xi) const
{
	size_t i, k;
	double sum;
	for (size_t n = 0; n < N - 1; ++n)
	{
		sum = 0.0;
		for (k = 0; k < K_; ++k)
			for (i = 0; i < K_; ++i)
			{
				//xi[n][k][i] = alpha[n][k] * beta[n+1][i] * A_[k][i] * B_[i][observations[n+1]];
				xi[n][k][i] = alpha[n][k] * beta[n+1][i] * A_[k][i] * evaluateEmissionProbability(i, observations[n+1]);
				sum += xi[n][k][i];
			}

		for (k = 0; k < K_; ++k)
			for (i = 0; i < K_; ++i)
				xi[n][k][i] /= sum;
	}
}

void DDHMM::generateSample(const size_t N, std::vector<unsigned int> &observations, std::vector<unsigned int> &states) const
{
	// PRECONDITIONS [] >>
	//	-. std::srand() had to be called before this function is called.

	states[0] = generateInitialState();
	observations[0] = generateObservationsSymbol(states[0]);

	for (size_t n = 1; n < N; ++n)
	{
		states[n] = generateNextState(states[n-1]);
		observations[n] = generateObservationsSymbol(states[n]);
	}
}

/*static*/ bool DDHMM::readSequence(std::istream &stream, size_t &N, std::vector<unsigned int> &observations)
{
	std::string dummy;

	stream >> dummy >> N;
#if defined(__GNUC__)
	if (strcasecmp(dummy.c_str(), "N=") != 0)
#elif defined(_MSC_VER)
	if (_stricmp(dummy.c_str(), "N=") != 0)
#endif
		return false;

	observations.resize(N);
	for (size_t n = 0; n < N; ++n)
		stream >> observations[n];

	return true;
}

/*static*/ bool DDHMM::writeSequence(std::ostream &stream, const std::vector<unsigned int> &observations)
{
	const size_t N = observations.size();
	stream << "N= " << N << std::endl;
	for (size_t n = 0; n < N; ++n)
		stream << observations[n] << std::endl;

	return true;
}

}  // namespace swl
