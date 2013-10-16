#include "swl/Config.h"
#include "swl/rnd_util/UnivariateNormalMixtureModel.h"
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/distributions/normal.hpp>  // for normal distribution
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <iostream>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

UnivariateNormalMixtureModel::UnivariateNormalMixtureModel(const size_t K)
: base_type(K, 1), mus_(K, 0.0), sigmas_(K, 0.0)
{
}

UnivariateNormalMixtureModel::UnivariateNormalMixtureModel(const size_t K, const std::vector<double> &pi, const dvector_type &mus, const dvector_type &sigmas)
: base_type(K, 1, pi), mus_(mus), sigmas_(sigmas)
{
}

UnivariateNormalMixtureModel::~UnivariateNormalMixtureModel()
{
}

void UnivariateNormalMixtureModel::doEstimateObservationDensityParametersByML(const size_t N, const unsigned int state, const dmatrix_type &observations, const dmatrix_type &gamma, const double sumGamma)
{
	// reestimate observation(emission) distribution in each state

	size_t n;

	// M-step
	double &mu = mus_[state];
	mu = 0.0;
	for (n = 0; n < N; ++n)
		mu += gamma(n, state) * observations(n, 0);
	mu = 0.001 + 0.999 * mu / sumGamma;

	//
	double &sigma = sigmas_[state];
	sigma = 0.0;
	for (n = 0; n < N; ++n)
		sigma += gamma(n, state) * (observations(n, 0) - mu) * (observations(n, 0) - mu);
	sigma = 0.001 + 0.999 * std::sqrt(sigma / sumGamma);
	assert(sigma > 0.0);

	// POSTCONDITIONS [] >>
	//	-. all standard deviations have to be positive.
}

void UnivariateNormalMixtureModel::doEstimateObservationDensityParametersByMAPUsingConjugatePrior(const size_t N, const unsigned int state, const dmatrix_type &observations, const dmatrix_type &gamma, const double sumGamma)
{
	throw std::runtime_error("not yet implemented");
}

double UnivariateNormalMixtureModel::doEvaluateMixtureComponentProbability(const unsigned int state, const boost::numeric::ublas::matrix_row<const dmatrix_type> &observation) const
{
	boost::math::normal pdf(mus_[state], sigmas_[state]);
	return boost::math::pdf(pdf, observation[0]);
}

void UnivariateNormalMixtureModel::doGenerateObservationsSymbol(const unsigned int state, boost::numeric::ublas::matrix_row<dmatrix_type> &observation, const unsigned int seed /*= (unsigned int)-1*/) const
{
	typedef boost::normal_distribution<> distribution_type;
	typedef boost::variate_generator<base_generator_type &, distribution_type> generator_type;

	if ((unsigned int)-1 != seed)
		baseGenerator_.seed(seed);

	generator_type normal_gen(baseGenerator_, distribution_type(mus_[state], sigmas_[state]));
	observation[0] = normal_gen();
}

bool UnivariateNormalMixtureModel::doReadObservationDensity(std::istream &stream)
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
	if (strcasecmp(dummy.c_str(), "normal") != 0)
#elif defined(_MSC_VER)
	if (_stricmp(dummy.c_str(), "normal") != 0)
#endif
		return false;

	stream >> dummy;
#if defined(__GNUC__)
	if (strcasecmp(dummy.c_str(), "mixture:") != 0)
#elif defined(_MSC_VER)
	if (_stricmp(dummy.c_str(), "mixture:") != 0)
#endif
		return false;

	stream >> dummy;
#if defined(__GNUC__)
	if (strcasecmp(dummy.c_str(), "mu:") != 0)
#elif defined(_MSC_VER)
	if (_stricmp(dummy.c_str(), "mu:") != 0)
#endif
		return false;

	// K
	for (size_t k = 0; k < K_; ++k)
		stream >> mus_[k];

	stream >> dummy;
#if defined(__GNUC__)
	if (strcasecmp(dummy.c_str(), "sigma:") != 0)
#elif defined(_MSC_VER)
	if (_stricmp(dummy.c_str(), "sigma:") != 0)
#endif
		return false;

	// K
	for (size_t k = 0; k < K_; ++k)
		stream >> sigmas_[k];

	return true;
}

bool UnivariateNormalMixtureModel::doWriteObservationDensity(std::ostream &stream) const
{
	stream << "univariate normal mixture:" << std::endl;

	// K
	stream << "mu:" << std::endl;
	for (size_t k = 0; k < K_; ++k)
		stream << mus_[k] << ' ';
	stream << std::endl;

	// K
	stream << "sigma:" << std::endl;
	for (size_t k = 0; k < K_; ++k)
		stream << sigmas_[k] << ' ';
	stream << std::endl;

	return true;
}

void UnivariateNormalMixtureModel::doInitializeObservationDensity(const std::vector<double> &lowerBoundsOfObservationDensity, const std::vector<double> &upperBoundsOfObservationDensity)
{
	// PRECONDITIONS [] >>
	//	-. std::srand() had to be called before this function is called.

	// initialize the parameters of observation density
	const std::size_t numLowerBound = lowerBoundsOfObservationDensity.size();
	const std::size_t numUpperBound = upperBoundsOfObservationDensity.size();

	const std::size_t numParameters = K_ * D_ * 2;  // the total number of parameters of observation density

	assert(numLowerBound == numUpperBound);
	assert(1 == numLowerBound || numParameters == numLowerBound);

	if (1 == numLowerBound)
	{
		const double lb = lowerBoundsOfObservationDensity[0], ub = upperBoundsOfObservationDensity[0];
		for (size_t k = 0; k < K_; ++k)
		{
			mus_[k] = ((double)std::rand() / RAND_MAX) * (ub - lb) + lb;
			sigmas_[k] = ((double)std::rand() / RAND_MAX) * (ub - lb) + lb;
		}
	}
	else if (numParameters == numLowerBound)
	{
		size_t k, idx = 0;
		for (k = 0; k < K_; ++k, ++idx)
			mus_[k] = ((double)std::rand() / RAND_MAX) * (upperBoundsOfObservationDensity[idx] - lowerBoundsOfObservationDensity[idx]) + lowerBoundsOfObservationDensity[idx];
		for (k = 0; k < K_; ++k, ++idx)
			sigmas_[k] = ((double)std::rand() / RAND_MAX) * (upperBoundsOfObservationDensity[idx] - lowerBoundsOfObservationDensity[idx]) + lowerBoundsOfObservationDensity[idx];
	}

	// POSTCONDITIONS [] >>
	//	-. all concentration parameters have to be greater than or equal to 0.
}

}  // namespace swl
