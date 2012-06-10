#include "swl/Config.h"
#include "swl/rnd_util/MixtureModel.h"
#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

MixtureModel::MixtureModel(const std::size_t K, const std::size_t D)
: K_(K), D_(D)
{
}

MixtureModel::MixtureModel(const std::size_t K, const std::size_t D, const std::vector<double> &pi)
: K_(K), D_(D), pi_(pi)
{
}

MixtureModel::~MixtureModel()
{
}

unsigned int MixtureModel::generateState() const
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

bool MixtureModel::readModel(std::istream &stream)
{
	std::string dummy;

	// TODO [check] >>
	std::size_t K;
	stream >> dummy >> K;  // the number of mixture components
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

	stream >> dummy;
#if defined(__GNUC__)
	if (strcasecmp(dummy.c_str(), "pi:") != 0)
#elif defined(_MSC_VER)
	if (_stricmp(dummy.c_str(), "pi:") != 0)
#endif
		return false;

	// K
	pi_.resize(K_);
	for (std::size_t k = 0; k < K_; ++k)
		stream >> pi_[k];

	return doReadObservationDensity(stream);
}

bool MixtureModel::writeModel(std::ostream &stream) const
{
	stream << "K= " << K_ << std::endl;  // the number of mixture components
	stream << "D= " << D_ << std::endl;  // the dimension of observation symbols

	// K
	stream << "pi:" << std::endl;
	for (std::size_t k = 0; k < K_; ++k)
		stream << pi_[k] << ' ';
	stream << std::endl;

	return doWriteObservationDensity(stream);
}

void MixtureModel::initializeModel(const std::vector<double> &lowerBoundsOfObservationDensity, const std::vector<double> &upperBoundsOfObservationDensity)
{
	// PRECONDITIONS [] >>
	//	-. std::srand() had to be called before this function is called.

	std::size_t k;
	double sum = 0.0;
	for (k = 0; k < K_; ++k)
	{
		pi_[k] = (double)std::rand() / RAND_MAX;
		sum += pi_[k];
	}
	for (k = 0; k < K_; ++k)
		pi_[k] /= sum;

	doInitializeObservationDensity(lowerBoundsOfObservationDensity, upperBoundsOfObservationDensity);
}

void MixtureModel::normalizeModelParameters()
{
	std::size_t k;
	double sum = 0.0;
	for (k = 0; k < K_; ++k)
		sum += pi_[k];
	for (k = 0; k < K_; ++k)
		pi_[k] /= sum;

	doNormalizeObservationDensityParameters();
}

}  // namespace swl
