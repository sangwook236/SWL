#include "swl/Config.h"
#include "swl/rnd_util/ArHmmWithUnivariateNormalObservations.h"
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

ArHmmWithUnivariateNormalObservations::ArHmmWithUnivariateNormalObservations(const size_t K, const size_t P)
: base_type(K, 1), mus_(K, 0.0), sigmas_(K, 0.0), Ws_(K, 0.0), P_(P),  // 0-based index.
  mus_conj_(), betas_conj_(), sigmas_conj_(), nus_conj_(),
  baseGenerator_()
{
}

ArHmmWithUnivariateNormalObservations::ArHmmWithUnivariateNormalObservations(const size_t K, const size_t P, const dvector_type &pi, const dmatrix_type &A, const dvector_type &mus, const dvector_type &sigmas, const dvector_type &Ws)
: base_type(K, 1, pi, A), mus_(mus), sigmas_(sigmas), Ws_(Ws), P_(P),
  mus_conj_(), betas_conj_(), sigmas_conj_(), nus_conj_(),
  baseGenerator_()
{
}

ArHmmWithUnivariateNormalObservations::ArHmmWithUnivariateNormalObservations(const size_t K, const size_t P, const dvector_type *pi_conj, const dmatrix_type *A_conj, const dvector_type *mus_conj, const dvector_type *betas_conj, const dvector_type *sigmas_conj, const dvector_type *nus_conj)
: base_type(K, 1, pi_conj, A_conj), mus_(K, 0.0), sigmas_(K, 0.0), Ws_(K, 0.0), P_(P),
  mus_conj_(mus_conj), betas_conj_(betas_conj), sigmas_conj_(sigmas_conj), nus_conj_(nus_conj),
  baseGenerator_()
{
	// FIXME [modify] >>
	throw std::runtime_error("not yet implemented");
}

ArHmmWithUnivariateNormalObservations::~ArHmmWithUnivariateNormalObservations()
{
}

void ArHmmWithUnivariateNormalObservations::doEstimateObservationDensityParametersByML(const size_t N, const unsigned int state, const dmatrix_type &observations, const dmatrix_type &gamma, const double denominatorA)
{
	// M-step.
	// reestimate observation(emission) distribution in each state.

	const double eps = 1e-50;
	size_t n;
	const double sumGamma = denominatorA + gamma(N-1, state);
	assert(std::fabs(sumGamma) >= eps);

	// TODO [check] >> assume x_-1 = observations(-1, 0) = 0.
	double denominatorW = 0.0;  // gamma(0, state) * observations(-1, 0) * observations(-1, 0).
	for (n = 1; n < N; ++n)
		denominatorW += gamma(n, state) * observations(n-1, 0) * observations(n-1, 0);
	assert(std::fabs(denominatorW) >= eps);

	//
	double &W = Ws_[state];
	double &mu = mus_[state];

	const double tol = 1.0e-5;
	const size_t maxIteration = 1000;
	size_t iteration = 0;
	double oldMu = mu, oldW = W;
	while (iteration < maxIteration)
	{
		// TODO [check] >> assume x_-1 = observations(-1, 0) = 0.
		W = 0.0;  // gamma(0, state) * (observations(0, 0) - mu) * observations(-1, 0) = 0.
		for (n = 1; n < N; ++n)
			W += gamma(n, state) * (observations(n, 0) - mu) * observations(n-1, 0);
		W = 0.001 + 0.999 * W / denominatorW;
	
		// TODO [check] >> assume x_-1 = observations(-1, 0) = 0.
		mu = gamma(0, state) * observations(0, 0);  // gamma(0, state) * (observations(0, 0) - W * observations(-1, 0)).
		for (n = 1; n < N; ++n)
			mu += gamma(n, state) * (observations(n, 0) - W * observations(n-1, 0));
		mu = 0.001 + 0.999 * mu / sumGamma;

		if (std::fabs(W - oldW) <= tol && std::fabs(mu - oldMu) <= tol)
			break;

		oldW = W;
		oldMu = mu;

		++iteration;
	}

	//
	double &sigma = sigmas_[state];
	// TODO [check] >> assume x_-1 = observations(-1, 0) = 0.
	sigma = gamma(0, state) * std::pow(observations(0, 0) - mu, 2.0);  // gamma(0, state) * (observations(0, 0) - W * observations(-1, 0) - mu)^2.
	for (n = 1; n < N; ++n)
		sigma += gamma(n, state) * std::pow(observations(n, 0) - W * observations(n-1, 0) - mu, 2.0);
	sigma = 0.001 + 0.999 * std::sqrt(sigma / sumGamma);
	assert(sigma > 0.0);

	// POSTCONDITIONS [] >>
	//	-. all standard deviations have to be positive.
}

void ArHmmWithUnivariateNormalObservations::doEstimateObservationDensityParametersByML(const std::vector<size_t> &Ns, const unsigned int state, const std::vector<dmatrix_type> &observationSequences, const std::vector<dmatrix_type> &gammas, const size_t R, const double denominatorA)
{
	// M-step.
	// reestimate observation(emission) distribution in each state.

	const double eps = 1e-50;
	size_t n, r;
	double sumGamma = denominatorA;
	for (r = 0; r < R; ++r)
		sumGamma += gammas[r](Ns[r]-1, state);
	assert(std::fabs(sumGamma) >= eps);

	double denominatorW = 0.0;
	for (r = 0; r < R; ++r)
	{
		const dmatrix_type &observationr = observationSequences[r];
		const dmatrix_type &gammar = gammas[r];

		// TODO [check] >> assume x_-1 = observations(-1, 0) = 0.
		//denominatorW += 0.0;  // gammar(0, state) * observationr(-1, 0) * observationr(-1, 0) = 0.
		for (n = 1; n < Ns[r]; ++n)
			denominatorW += gammar(n, state) * observationr(n-1, 0) * observationr(n-1, 0);
	}
	assert(std::fabs(denominatorW) >= eps);

	//
	double &W = Ws_[state];
	double &mu = mus_[state];

	const double tol = 1.0e-5;
	const size_t maxIteration = 1000;
	size_t iteration = 0;
	double oldMu = mu, oldW = W;
	while (iteration < maxIteration)
	{
		W = 0.0;
		for (r = 0; r < R; ++r)
		{
			const dmatrix_type &observationr = observationSequences[r];
			const dmatrix_type &gammar = gammas[r];

			// TODO [check] >> assume x_-1 = observationr(-1, 0) = 0.
			//W += 0.0;  // gammar(0, state) * (observationr(0, 0) - mu) * observationr(-1, 0) = 0.
			for (n = 1; n < Ns[r]; ++n)
				W += gammar(n, state) * (observationr(n, 0) - mu) * observationr(n-1, 0);
		}
		W = 0.001 + 0.999 * W / denominatorW;

		mu = 0.0;
		for (r = 0; r < R; ++r)
		{
			const dmatrix_type &observationr = observationSequences[r];
			const dmatrix_type &gammar = gammas[r];
	
			// TODO [check] >> assume x_-1 = observationr(-1, 0) = 0.
			mu += gammar(0, state) * observationr(0, 0);  // gammar(0, state) * (observationr(0, 0) - W * observationr(-1, 0)).
			for (n = 1; n < Ns[r]; ++n)
				mu += gammar(n, state) * (observationr(n, 0) - W * observationr(n-1, 0));
		}
		mu = 0.001 + 0.999 * mu / sumGamma;

		if (std::fabs(W - oldW) <= tol && std::fabs(mu - oldMu) <= tol)
			break;

		oldW = W;
		oldMu = mu;

		++iteration;
	}

	//
	double &sigma = sigmas_[state];
	sigma = 0.0;
	for (r = 0; r < R; ++r)
	{
		const dmatrix_type &observationr = observationSequences[r];
		const dmatrix_type &gammar = gammas[r];

		// TODO [check] >> assume x_-1 = observationr(-1, 0) = 0.
		sigma += gammar(0, state) * std::pow(observationr(0, 0) - mu, 2.0);  // gammar(0, state) * (observationr(0, 0) - W * observationr(-1, 0) - mu)^2.
		for (n = 1; n < Ns[r]; ++n)
			sigma += gammar(n, state) * std::pow(observationr(n, 0) - W * observationr(n-1, 0) - mu, 2.0);
	}
	sigma = 0.001 + 0.999 * std::sqrt(sigma / sumGamma);
	assert(sigma > 0.0);

	// POSTCONDITIONS [] >>
	//	-. all standard deviations have to be positive.
}

void ArHmmWithUnivariateNormalObservations::doEstimateObservationDensityParametersByMAPUsingConjugatePrior(const size_t N, const unsigned int state, const dmatrix_type &observations, const dmatrix_type &gamma, const double denominatorA)
{
	// FIXME [modify] >>
	throw std::runtime_error("not yet implemented");

	// M-step.
	// reestimate observation(emission) distribution in each state.

	const double eps = 1e-50;
	size_t n;
	const double sumGamma = denominatorA + gamma(N-1, state);
	assert(std::fabs(sumGamma) >= eps);

	//
	double &mu = mus_[state];
	mu = (*betas_conj_)(state) * (*mus_conj_)(state);
	for (n = 0; n < N; ++n)
		mu += gamma(n, state) * observations(n, 0);
	mu = 0.001 + 0.999 * mu / (sumGamma + (*betas_conj_)(state));

	//
	double &sigma = sigmas_[state];
	sigma = (*sigmas_conj_)(state) + (*betas_conj_)(state) * (mu - (*mus_conj_)(state)) * (mu - (*mus_conj_)(state));
	for (n = 0; n < N; ++n)
		sigma += gamma(n, state) * (observations(n, 0) - mu) * (observations(n, 0) - mu);
	sigma = 0.001 + 0.999 * std::sqrt(sigma / (sumGamma + (*nus_conj_)(state) - D_));
	assert(sigma > 0.0);

	// POSTCONDITIONS [] >>
	//	-. all standard deviations have to be positive.
}

void ArHmmWithUnivariateNormalObservations::doEstimateObservationDensityParametersByMAPUsingConjugatePrior(const std::vector<size_t> &Ns, const unsigned int state, const std::vector<dmatrix_type> &observationSequences, const std::vector<dmatrix_type> &gammas, const size_t R, const double denominatorA)
{
	// FIXME [modify] >>
	throw std::runtime_error("not yet implemented");

	// M-step.
	// reestimate observation(emission) distribution in each state.

	const double eps = 1e-50;
	size_t n, r;
	double sumGamma = denominatorA;
	for (r = 0; r < R; ++r)
		sumGamma += gammas[r](Ns[r]-1, state);
	assert(std::fabs(sumGamma) >= eps);

	//
	double &mu = mus_[state];
	mu = (*betas_conj_)(state) * (*mus_conj_)(state);
	for (r = 0; r < R; ++r)
	{
		const dmatrix_type &observationr = observationSequences[r];
		const dmatrix_type &gammar = gammas[r];

		for (n = 0; n < Ns[r]; ++n)
			mu += gammar(n, state) * observationr(n, 0);
	}
	mu = 0.001 + 0.999 * mu / (sumGamma + (*betas_conj_)(state));

	//
	double &sigma = sigmas_[state];
	sigma = (*sigmas_conj_)(state) + (*betas_conj_)(state) * (mu - (*mus_conj_)(state)) * (mu - (*mus_conj_)(state));
	for (r = 0; r < R; ++r)
	{
		const dmatrix_type &observationr = observationSequences[r];
		const dmatrix_type &gammar = gammas[r];

		for (n = 0; n < Ns[r]; ++n)
			sigma += gammar(n, state) * (observationr(n, 0) - mu) * (observationr(n, 0) - mu);
	}
	sigma = 0.001 + 0.999 * std::sqrt(sigma / (sumGamma + (*nus_conj_)(state) - D_));
	assert(sigma > 0.0);

	// POSTCONDITIONS [] >>
	//	-. all standard deviations have to be positive.
}

void ArHmmWithUnivariateNormalObservations::doEstimateObservationDensityParametersByMAPUsingEntropicPrior(const size_t N, const unsigned int state, const dmatrix_type &observations, const dmatrix_type &gamma, const double /*z*/, const bool /*doesTrimParameter*/, const double /*terminationTolerance*/, const size_t /*maxIteration*/, const double denominatorA)
{
	doEstimateObservationDensityParametersByML(N, state, observations, gamma, denominatorA);
}

void ArHmmWithUnivariateNormalObservations::doEstimateObservationDensityParametersByMAPUsingEntropicPrior(const std::vector<size_t> &Ns, const unsigned int state, const std::vector<dmatrix_type> &observationSequences, const std::vector<dmatrix_type> &gammas, const double /*z*/, const bool /*doesTrimParameter*/, const double /*terminationTolerance*/, const size_t /*maxIteration*/, const size_t R, const double denominatorA)
{
	doEstimateObservationDensityParametersByML(Ns, state, observationSequences, gammas, R, denominatorA);
}

double ArHmmWithUnivariateNormalObservations::doEvaluateEmissionProbability(const unsigned int state, const dvector_type &observation) const
{
	assert(false);

	//boost::math::normal pdf;  // (default mean = zero, and standard deviation = unity).
	boost::math::normal pdf(mus_[state], sigmas_[state]);

	return boost::math::pdf(pdf, observation[0]);
}

double ArHmmWithUnivariateNormalObservations::doEvaluateEmissionProbability(const unsigned int state, const size_t n, const dmatrix_type &observations) const
{
	//boost::math::normal pdf;  // (default mean = zero, and standard deviation = unity).
	// TODO [check] >> assume x_-1 = observations(-1, 0) = 0.
	boost::math::normal pdf(0 == n ? mus_[state] : (Ws_[state] * observations(n-1, 0) + mus_[state]), sigmas_[state]);

	return observations(n, 0);
}

void ArHmmWithUnivariateNormalObservations::doGenerateObservationsSymbol(const unsigned int state, const size_t n, dmatrix_type &observations) const
{
	typedef boost::normal_distribution<> distribution_type;
	typedef boost::variate_generator<base_generator_type &, distribution_type> generator_type;

	// TODO [check] >> assume x_-1 = observations(-1, 0) = 0.
	generator_type normal_gen(baseGenerator_, distribution_type(0 == n ? mus_[state] : (Ws_[state] * observations(n-1, 0) + mus_[state]), sigmas_[state]));
	observations(n, 0) = normal_gen();
}

void ArHmmWithUnivariateNormalObservations::doInitializeRandomSampleGeneration(const unsigned int seed /*= (unsigned int)-1*/) const
{
	if ((unsigned int)-1 != seed)
		baseGenerator_.seed(seed);
}

bool ArHmmWithUnivariateNormalObservations::doReadObservationDensity(std::istream &stream)
{
	if (1 != D_) return false;

	std::string dummy;
	stream >> dummy;
#if defined(__GNUC__)
	if (strcasecmp(dummy.c_str(), "ar") != 0)
#elif defined(_MSC_VER)
	if (_stricmp(dummy.c_str(), "ar") != 0)
#endif
		return false;

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
	if (strcasecmp(dummy.c_str(), "W:") != 0)
#elif defined(_MSC_VER)
	if (_stricmp(dummy.c_str(), "W:") != 0)
#endif
		return false;

	// K.
	for (size_t k = 0; k < K_; ++k)
		stream >> Ws_[k];

	stream >> dummy;
#if defined(__GNUC__)
	if (strcasecmp(dummy.c_str(), "mu:") != 0)
#elif defined(_MSC_VER)
	if (_stricmp(dummy.c_str(), "mu:") != 0)
#endif
		return false;

	// K.
	for (size_t k = 0; k < K_; ++k)
		stream >> mus_[k];

	stream >> dummy;
#if defined(__GNUC__)
	if (strcasecmp(dummy.c_str(), "sigma:") != 0)
#elif defined(_MSC_VER)
	if (_stricmp(dummy.c_str(), "sigma:") != 0)
#endif
		return false;

	// K.
	for (size_t k = 0; k < K_; ++k)
		stream >> sigmas_[k];

	return true;
}

bool ArHmmWithUnivariateNormalObservations::doWriteObservationDensity(std::ostream &stream) const
{
	stream << "ar univariate normal:" << std::endl;

	// K.
	stream << "W:" << std::endl;
	for (size_t k = 0; k < K_; ++k)
		stream << Ws_[k] << ' ';
	stream << std::endl;

	// K.
	stream << "mu:" << std::endl;
	for (size_t k = 0; k < K_; ++k)
		stream << mus_[k] << ' ';
	stream << std::endl;

	// K.
	stream << "sigma:" << std::endl;
	for (size_t k = 0; k < K_; ++k)
		stream << sigmas_[k] << ' ';
	stream << std::endl;

	return true;
}

void ArHmmWithUnivariateNormalObservations::doInitializeObservationDensity(const std::vector<double> &lowerBoundsOfObservationDensity, const std::vector<double> &upperBoundsOfObservationDensity)
{
	// PRECONDITIONS [] >>
	//	-. std::srand() has to be called before this function is called.

	// initialize the parameters of observation density.
	const std::size_t numLowerBound = lowerBoundsOfObservationDensity.size();
	const std::size_t numUpperBound = upperBoundsOfObservationDensity.size();

	const std::size_t numParameters = K_ * D_ * 3;  // the total number of parameters of observation density.

	assert(numLowerBound == numUpperBound);
	assert(1 == numLowerBound || numParameters == numLowerBound);

	if (1 == numLowerBound)
	{
		const double lb = lowerBoundsOfObservationDensity[0], ub = upperBoundsOfObservationDensity[0];
		for (size_t k = 0; k < K_; ++k)
		{
			Ws_[k] = ((double)std::rand() / RAND_MAX) * (ub - lb) + lb;
			mus_[k] = ((double)std::rand() / RAND_MAX) * (ub - lb) + lb;
			// TODO [check] >> all standard deviations have to be positive.
			sigmas_[k] = ((double)std::rand() / RAND_MAX) * (ub - lb) + lb;
		}
	}
	else if (numParameters == numLowerBound)
	{
		size_t k, idx = 0;
		for (k = 0; k < K_; ++k, ++idx)
			Ws_[k] = ((double)std::rand() / RAND_MAX) * (upperBoundsOfObservationDensity[idx] - lowerBoundsOfObservationDensity[idx]) + lowerBoundsOfObservationDensity[idx];
		for (k = 0; k < K_; ++k, ++idx)
			mus_[k] = ((double)std::rand() / RAND_MAX) * (upperBoundsOfObservationDensity[idx] - lowerBoundsOfObservationDensity[idx]) + lowerBoundsOfObservationDensity[idx];
		for (k = 0; k < K_; ++k, ++idx)
			// TODO [check] >> all standard deviations have to be positive.
			sigmas_[k] = ((double)std::rand() / RAND_MAX) * (upperBoundsOfObservationDensity[idx] - lowerBoundsOfObservationDensity[idx]) + lowerBoundsOfObservationDensity[idx];
	}

	for (size_t k = 0; k < K_; ++k)
	{
		// all standard deviations have to be positive.
		if (sigmas_[k] < 0.0)
			sigmas_[k] = -sigmas_[k];
	}

	// POSTCONDITIONS [] >>
	//	-. all standard deviations have to be positive.
}

}  // namespace swl
