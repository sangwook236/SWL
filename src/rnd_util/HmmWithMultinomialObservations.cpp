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

void HmmWithMultinomialObservations::doEstimateObservationDensityParametersInMStep(const size_t N, const unsigned int state, const std::vector<unsigned int> &observations, const boost::multi_array<double, 2> &gamma, const double denominatorA)
{
	size_t n;

	// reestimate symbol prob in each state
	const double denominatorB = denominatorA + gamma[N-1][state];
	double numeratorB;
	for (size_t d = 0; d < D_; ++d)
	{
		numeratorB = 0.0;
		for (n = 0; n < N; ++n)
		{
			if (observations[n] == (unsigned int)d)
				numeratorB += gamma[n][state];
		}

		B_[state][d] = 0.001 + 0.999 * numeratorB / denominatorB;
	}
}

void HmmWithMultinomialObservations::doEstimateObservationDensityParametersInMStep(const std::vector<size_t> &Ns, const unsigned int state, const std::vector<std::vector<unsigned int> > &observationSequences, const std::vector<boost::multi_array<double, 2> > &gammas, const size_t R, const double denominatorA)
{
	size_t n, r;

	// reestimate symbol prob in each state
	double denominatorB = denominatorA;
	for (r = 0; r < R; ++r)
		denominatorB += gammas[r][Ns[r]-1][state];

	double numeratorB;
	for (size_t d = 0; d < D_; ++d)
	{
		numeratorB = 0.0;
		for (r = 0; r < R; ++r)
			for (n = 0; n < Ns[r]; ++n)
			{
				if (observationSequences[r][n] == (unsigned int)d)
					numeratorB += gammas[r][n][state];
			}

		B_[state][d] = 0.001 + 0.999 * numeratorB / denominatorB;
	}
}

unsigned int HmmWithMultinomialObservations::doGenerateObservationsSymbol(const unsigned int state) const
{
	// PRECONDITIONS [] >>
	//	-. std::srand() had to be called before this function is called.

	const double prob = (double)std::rand() / RAND_MAX;

	double accum = 0.0;
	unsigned int observation = (unsigned int)D_;
	for (size_t d = 0; d < D_; ++d)
	{
		accum += B_[state][d];
		//accum += doEvaluateEmissionProbability(state, d);
		if (prob < accum)
		{
			observation = (unsigned int)d;
			break;
		}
	}

	return observation;

	// POSTCONDITIONS [] >>
	//	-. if observation = D_, an error occurs.
}

bool HmmWithMultinomialObservations::doReadObservationDensity(std::istream &stream)
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

bool HmmWithMultinomialObservations::doWriteObservationDensity(std::ostream &stream) const
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

void HmmWithMultinomialObservations::doInitializeObservationDensity()
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

void HmmWithMultinomialObservations::doNormalizeObservationDensityParameters()
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
