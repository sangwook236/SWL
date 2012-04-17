#include "swl/Config.h"
#include "swl/rnd_util/HmmWithMultinomialObservations.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

HmmWithMultinomialObservations::HmmWithMultinomialObservations(const size_t K, const size_t D)
: base_type(K, D), B_(boost::extents[K][D])  // 0-based index
//: base_type(K, D), B_(boost::extents[boost::multi_array_types::extent_range(1, K+1)][boost::multi_array_types::extent_range(1, D+1)])  // 1-based index
{
}

HmmWithMultinomialObservations::HmmWithMultinomialObservations(const size_t K, const size_t D, const std::vector<double> &pi, const boost::multi_array<double, 2> &A, const boost::multi_array<double, 2> &B)
: base_type(K, D, pi, A), B_(B)
{
}

HmmWithMultinomialObservations::~HmmWithMultinomialObservations()
{
}

bool HmmWithMultinomialObservations::estimateParameters(const size_t N, const std::vector<unsigned int> &observations, const double terminationTolerance, boost::multi_array<double, 2> &alpha, boost::multi_array<double, 2> &beta, boost::multi_array<double, 2> &gamma, size_t &numIteration, double &initLogProbability, double &finalLogProbability)
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
	double numeratorB, denominatorB;
	double delta, logprobprev = logprobf;
	size_t i, k, n;
	size_t iter = 0;
	do
	{
		for (k = 0; k < K_; ++k)
		{
			// reestimate frequency of state k in time n=0
			pi_[k] = .001 + .999 * gamma[0][k];

			// reestimate transition matrix in each state
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
			denominatorB = denominatorA + gamma[N-1][k];

			for (i = 0; i < D_; ++i)
			{
				numeratorB = 0.0;
				for (n = 0; n < N; ++n)
				{
					if (observations[n] == (unsigned int)i)
						numeratorB += gamma[n][k];
				}

				B_[k][i] = .001 + .999 * numeratorB / denominatorB;
			}
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
	finalLogProbability = logprobf;  // log P(O | estimated model)

	return true;
}

unsigned int HmmWithMultinomialObservations::generateObservationsSymbol(const unsigned int state) const
{
	// PRECONDITIONS [] >>
	//	-. std::srand() had to be called before this function is called.

	const double prob = (double)std::rand() / RAND_MAX;

	double accum = 0.0;
	unsigned int observation = (unsigned int)D_;
	for (size_t i = 0; i < D_; ++i)
	{
		accum += B_[state][i];
		//accum += evaluateEmissionProbability(state, i);
		if (prob < accum)
		{
			observation = (unsigned int)i;
			break;
		}
	}

	return observation;

	// POSTCONDITIONS [] >>
	//	-. if observation = D_, an error occurs.
}

bool HmmWithMultinomialObservations::readObservationDensity(std::istream &stream)
{
	size_t i, k;
	std::string dummy;

	stream >> dummy;
#if defined(__GNUC__)
	if (strcasecmp(dummy.c_str(), "B:") != 0)
#elif defined(_MSC_VER)
	if (_stricmp(dummy.c_str(), "B:") != 0)
#endif
		return false;

	B_.resize(boost::extents[K_][D_]);;
	for (k = 0; k < K_; ++k)
		for (i = 0; i < D_; ++i)
			stream >> B_[k][i];

	return true;
}

bool HmmWithMultinomialObservations::writeObservationDensity(std::ostream &stream) const
{
	size_t i, k;

	stream << "B:" << std::endl;
	for (k = 0; k < K_; ++k)
	{
		for (i = 0; i < D_; ++i)
			stream << B_[k][i] << ' ';
		stream << std::endl;
	}

	return true;
}

void HmmWithMultinomialObservations::initializeObservationDensity()
{
	// PRECONDITIONS [] >>
	//	-. std::srand() had to be called before this function is called.

	size_t i;
	double sum;
	for (size_t k = 0; k < K_; ++k)
	{
		sum = 0.0;
		for (i = 0; i < D_; ++i)
		{
			B_[k][i] = (double)std::rand() / RAND_MAX;
			sum += B_[k][i];
		}
		for (i = 0; i < D_; ++i)
			B_[k][i] /= sum;
	}
}

void HmmWithMultinomialObservations::normalizeObservationDensityParameters()
{
	size_t i;
	double sum;

	for (size_t k = 0; k < K_; ++k)
	{
		sum = 0.0;
		for (i = 0; i < D_; ++i)
			sum += B_[k][i];
		for (i = 0; i < D_; ++i)
			B_[k][i] /= sum;
	}
}

}  // namespace swl
