#include "swl/Config.h"
#include "swl/rnd_util/HMM.h"
#include <boost/math/constants/constants.hpp>
#include <cstring>
#include <cstdlib>
#include <cassert>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

HMM::HMM(const std::size_t K, const std::size_t D)
: K_(K), D_(D), pi_(K, 0.0), A_(K, K, 0.0),  // 0-based index
  pi_conj_(), A_conj_()
{
}

HMM::HMM(const std::size_t K, const std::size_t D, const dvector_type &pi, const dmatrix_type &A)
: K_(K), D_(D), pi_(pi), A_(A),
  pi_conj_(), A_conj_()
{
}

HMM::HMM(const std::size_t K, const std::size_t D, const dvector_type *pi_conj, const dmatrix_type *A_conj)
: K_(K), D_(D), pi_(K, 0.0), A_(K, K, 0.0),
  pi_conj_(pi_conj), A_conj_(A_conj)
{
}

HMM::~HMM()
{
}

void HMM::computeGamma(const std::size_t N, const dmatrix_type &alpha, const dmatrix_type &beta, dmatrix_type &gamma) const
{
	std::size_t k;
	double denominator;
	for (std::size_t n = 0; n < N; ++n)
	{
		denominator = 0.0;
		for (k = 0; k < K_; ++k)
		{
			gamma(n, k) = alpha(n, k) * beta(n, k);
			denominator += gamma(n, k);
		}

		for (k = 0; k < K_; ++k)
			gamma(n, k) /= denominator;
	}
}

unsigned int HMM::generateInitialState() const
{
	const double prob = (double)std::rand() / RAND_MAX;

	double accum = 0.0;
	unsigned int state = (unsigned int)K_;
	for (std::size_t k = 0; k < K_; ++k)
	{
		accum += pi_[k];
		if (prob < accum)
		{
			state = (unsigned int)k;
			break;
		}
	}

	// TODO [check] >>
	if ((unsigned int)K_ == state)
		state = (unsigned int)(K_ - 1);

	return state;

	// POSTCONDITIONS [] >>
	//	-. if state = K_, an error occurs.
}

unsigned int HMM::generateNextState(const unsigned int currState) const
{
	const double prob = (double)std::rand() / RAND_MAX;

	double accum = 0.0;
	unsigned int nextState = (unsigned int)K_;
	for (std::size_t k = 0; k < K_; ++k)
	{
		accum += A_(currState, k);
		if (prob < accum)
		{
			nextState = (unsigned int)k;
			break;
		}
	}

	// TODO [check] >>
	if ((unsigned int)K_ == nextState)
		nextState = (unsigned int)(K_ - 1);

	return nextState;

	// POSTCONDITIONS [] >>
	//	-. if nextState = K_, an error occurs.
}

bool HMM::readModel(std::istream &stream)
{
	std::string dummy;

	// TODO [check] >>
	std::size_t K;
	stream >> dummy >> K;  // the dimension of hidden states
#if defined(__GNUC__)
	if (strcasecmp(dummy.c_str(), "K=") != 0 || K_ != K)
#elif defined(_MSC_VER)
	if (_stricmp(dummy.c_str(), "K=") != 0 || K_ != K)
#endif
		return false;

	std::size_t D;
	stream >> dummy >> D;  // the dimension of observation symbols
#if defined(__GNUC__)
	if (strcasecmp(dummy.c_str(), "D=") != 0 || D_ != D)
#elif defined(_MSC_VER)
	if (_stricmp(dummy.c_str(), "D=") != 0 || D_ != D)
#endif
		return false;

	std::size_t i, k;
	stream >> dummy;
#if defined(__GNUC__)
	if (strcasecmp(dummy.c_str(), "pi:") != 0)
#elif defined(_MSC_VER)
	if (_stricmp(dummy.c_str(), "pi:") != 0)
#endif
		return false;

	// K
	pi_.resize(K_);
	for (k = 0; k < K_; ++k)
		stream >> pi_[k];

	stream >> dummy;
#if defined(__GNUC__)
	if (strcasecmp(dummy.c_str(), "A:") != 0)
#elif defined(_MSC_VER)
	if (_stricmp(dummy.c_str(), "A:") != 0)
#endif
		return false;

	// K x K
	A_.resize(K_, K_);
	for (k = 0; k < K_; ++k)
		for (i = 0; i < K_; ++i)
			stream >> A_(k, i);

	return doReadObservationDensity(stream);
}

bool HMM::writeModel(std::ostream &stream) const
{
	std::size_t i, k;

	stream << "K= " << K_ << std::endl;  // the dimension of hidden states
	stream << "D= " << D_ << std::endl;  // the dimension of observation symbols

	// K
	stream << "pi:" << std::endl;
	for (k = 0; k < K_; ++k)
		stream << pi_[k] << ' ';
	stream << std::endl;

	// K x K
	stream << "A:" << std::endl;
	for (k = 0; k < K_; ++k)
	{
		for (i = 0; i < K_; ++i)
			stream << A_(k, i) << ' ';
		stream << std::endl;
	}

	return doWriteObservationDensity(stream);
}

void HMM::initializeModel(const std::vector<double> &lowerBoundsOfObservationDensity, const std::vector<double> &upperBoundsOfObservationDensity)
{
	// PRECONDITIONS [] >>
	//	-. std::srand() had to be called before this function is called.

	std::size_t i, k;
	double sum = 0.0;
	for (k = 0; k < K_; ++k)
	{
		pi_[k] = (double)std::rand() / RAND_MAX;
		sum += pi_[k];
	}
	for (k = 0; k < K_; ++k)
		pi_[k] /= sum;

	for (k = 0; k < K_; ++k)
	{
		sum = 0.0;
		for (i = 0; i < K_; ++i)
		{
			A_(k, i) = (double)std::rand() / RAND_MAX;
			sum += A_(k, i);
		}
		for (i = 0; i < K_; ++i)
			A_(k, i) /= sum;
	}

	doInitializeObservationDensity(lowerBoundsOfObservationDensity, upperBoundsOfObservationDensity);
}

void HMM::normalizeModelParameters()
{
	std::size_t i, k;
	double sum;

	sum = 0.0;
	for (k = 0; k < K_; ++k)
		sum += pi_[k];
	for (k = 0; k < K_; ++k)
		pi_[k] /= sum;

	for (k = 0; k < K_; ++k)
	{
		sum = 0.0;
		for (i = 0; i < K_; ++i)
			sum += A_(k, i);
		for (i = 0; i < K_; ++i)
			A_(k, i) /= sum;
	}

	doNormalizeObservationDensityParameters();
}

}  // namespace swl
