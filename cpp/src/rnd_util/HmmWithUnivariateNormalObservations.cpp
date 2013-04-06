#include "swl/Config.h"
#include "swl/rnd_util/HmmWithUnivariateNormalObservations.h"
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/math/distributions/normal.hpp>  // for normal distribution
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <ctime>
#include <stdexcept>


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

HmmWithUnivariateNormalObservations::HmmWithUnivariateNormalObservations(const size_t K, const dvector_type &pi, const dmatrix_type &A, const dvector_type &mus, const dvector_type &sigmas)
: base_type(K, 1, pi, A), mus_(mus), sigmas_(sigmas),
  baseGenerator_()
{
}

HmmWithUnivariateNormalObservations::~HmmWithUnivariateNormalObservations()
{
}

void HmmWithUnivariateNormalObservations::doEstimateObservationDensityParametersByML(const size_t N, const unsigned int state, const dmatrix_type &observations, dmatrix_type &gamma, const double denominatorA)
{
	// reestimate observation(emission) distribution in each state

	size_t n;
	const double denominator = denominatorA + gamma(N-1, state);

	//
	double &mu = mus_[state];
	mu = 0.0;
	for (n = 0; n < N; ++n)
		mu += gamma(n, state) * observations(n, 0);
	mu = 0.001 + 0.999 * mu / denominator;

	//
	double &sigma = sigmas_[state];
	sigma = 0.0;
	for (n = 0; n < N; ++n)
		sigma += gamma(n, state) * (observations(n, 0) - mu) * (observations(n, 0) - mu);
	sigma = 0.001 + 0.999 * std::sqrt(sigma / denominator);
	assert(sigma > 0.0);

	// POSTCONDITIONS [] >>
	//	-. all standard deviations have to be positive.
}

void HmmWithUnivariateNormalObservations::doEstimateObservationDensityParametersByML(const std::vector<size_t> &Ns, const unsigned int state, const std::vector<dmatrix_type> &observationSequences, const std::vector<dmatrix_type> &gammas, const size_t R, const double denominatorA)
{
	// reestimate observation(emission) distribution in each state

	size_t n, r;
	double denominator = denominatorA;
	for (r = 0; r < R; ++r)
		denominator += gammas[r](Ns[r]-1, state);

	//
	double &mu = mus_[state];
	mu = 0.0;
	for (r = 0; r < R; ++r)
	{
		const dmatrix_type &observationr = observationSequences[r];
		const dmatrix_type &gammar = gammas[r];

		for (n = 0; n < Ns[r]; ++n)
			mu += gammar(n, state) * observationr(n, 0);
	}
	mu = 0.001 + 0.999 * mu / denominator;

	//
	double &sigma = sigmas_[state];
	sigma = 0.0;
	for (r = 0; r < R; ++r)
	{
		const dmatrix_type &observationr = observationSequences[r];
		const dmatrix_type &gammar = gammas[r];

		for (n = 0; n < Ns[r]; ++n)
			sigma += gammar(n, state) * (observationr(n, 0) - mu) * (observationr(n, 0) - mu);
	}
	sigma = 0.001 + 0.999 * std::sqrt(sigma / denominator);
	assert(sigma > 0.0);

	// POSTCONDITIONS [] >>
	//	-. all standard deviations have to be positive.
}

void HmmWithUnivariateNormalObservations::doEstimateObservationDensityParametersByMAP(const size_t N, const unsigned int state, const dmatrix_type &observations, dmatrix_type &gamma, const double denominatorA)
{
	throw std::runtime_error("not yet implemented");
}

void HmmWithUnivariateNormalObservations::doEstimateObservationDensityParametersByMAP(const std::vector<size_t> &Ns, const unsigned int state, const std::vector<dmatrix_type> &observationSequences, const std::vector<dmatrix_type> &gammas, const size_t R, const double denominatorA)
{
	throw std::runtime_error("not yet implemented");
}

double HmmWithUnivariateNormalObservations::doEvaluateEmissionProbability(const unsigned int state, const boost::numeric::ublas::matrix_row<const dmatrix_type> &observation) const
{
	//boost::math::normal pdf;  // (default mean = zero, and standard deviation = unity)
	boost::math::normal pdf(mus_[state], sigmas_[state]);

	return boost::math::pdf(pdf, observation[0]);
}

void HmmWithUnivariateNormalObservations::doGenerateObservationsSymbol(const unsigned int state, boost::numeric::ublas::matrix_row<dmatrix_type> &observation, const unsigned int seed /*= (unsigned int)-1*/) const
{
	typedef boost::normal_distribution<> distribution_type;
	typedef boost::variate_generator<base_generator_type &, distribution_type> generator_type;

	if ((unsigned int)-1 != seed)
		baseGenerator_.seed(seed);

	generator_type normal_gen(baseGenerator_, distribution_type(mus_[state], sigmas_[state]));
	observation[0] = normal_gen();
}

bool HmmWithUnivariateNormalObservations::doReadObservationDensity(std::istream &stream)
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

bool HmmWithUnivariateNormalObservations::doWriteObservationDensity(std::ostream &stream) const
{
	stream << "univariate normal:" << std::endl;

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

void HmmWithUnivariateNormalObservations::doInitializeObservationDensity(const std::vector<double> &lowerBoundsOfObservationDensity, const std::vector<double> &upperBoundsOfObservationDensity)
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
			// TODO [check] >> all standard deviations have to be positive.
			sigmas_[k] = ((double)std::rand() / RAND_MAX) * (ub - lb) + lb;
		}
	}
	else if (numParameters == numLowerBound)
	{
		size_t k, idx = 0;
		for (k = 0; k < K_; ++k, ++idx)
			mus_[k] = ((double)std::rand() / RAND_MAX) * (upperBoundsOfObservationDensity[idx] - lowerBoundsOfObservationDensity[idx]) + lowerBoundsOfObservationDensity[idx];
		for (k = 0; k < K_; ++k, ++idx)
			// TODO [check] >> all standard deviations have to be positive.
			sigmas_[k] = ((double)std::rand() / RAND_MAX) * (upperBoundsOfObservationDensity[idx] - lowerBoundsOfObservationDensity[idx]) + lowerBoundsOfObservationDensity[idx];
	}

	// POSTCONDITIONS [] >>
	//	-. all standard deviations have to be positive.
}

}  // namespace swl