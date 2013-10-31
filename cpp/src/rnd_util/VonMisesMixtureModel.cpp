#include "swl/Config.h"
#include "swl/rnd_util/VonMisesMixtureModel.h"
#include "swl/math/MathConstant.h"
#include "RndUtilLocalApi.h"
#include "swl/rnd_util/RejectionSampling.h"
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/math/constants/constants.hpp>
#include <iostream>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

// [ref] swl/src/rnd_util/RndUtilLocalApi.cpp
bool one_dim_root_finding_using_f(const double A, const double lower, const double upper, const std::size_t maxIteration, double &kappa);
double evaluateVonMisesDistribution(const double x, const double mu, const double kappa);

VonMisesMixtureModel::VonMisesMixtureModel(const size_t K)
: base_type(K, 1), mus_(K, 0.0), kappas_(K, 0.0),
  ms_conj_(), Rs_conj_(), cs_conj_(),
  targetDist_(), proposalDist_()
{
}

VonMisesMixtureModel::VonMisesMixtureModel(const size_t K, const std::vector<double> &pi, const dvector_type &mus, const dvector_type &kappas)
: base_type(K, 1, pi), mus_(mus), kappas_(kappas),
  ms_conj_(), Rs_conj_(), cs_conj_(),
  targetDist_(), proposalDist_()
{
}

VonMisesMixtureModel::VonMisesMixtureModel(const size_t K, const std::vector<double> *pi_conj, const dvector_type *ms_conj, const dvector_type *Rs_conj, const dvector_type *cs_conj)
: base_type(K, 1, pi_conj), mus_(K, 0.0), kappas_(K, 0.0),
  ms_conj_(ms_conj), Rs_conj_(Rs_conj), cs_conj_(cs_conj),
  targetDist_(), proposalDist_()
{
}

VonMisesMixtureModel::~VonMisesMixtureModel()
{
}

void VonMisesMixtureModel::doEstimateObservationDensityParametersByML(const size_t N, const unsigned int state, const dmatrix_type &observations, const dmatrix_type &gamma, const double sumGamma)
{
	// M-step.
	// reestimate observation(emission) distribution in each state.

	size_t n;

	double numerator = 0.0, denominator = 0.0;
	for (n = 0; n < N; ++n)
	{
		numerator += gamma(n, state) * std::sin(observations(n, 0));
		denominator += gamma(n, state) * std::cos(observations(n, 0));
	}

	double &mu = mus_[state];

	// TODO [check] >> check the range of each mu, [0, 2 * pi).
#if 0
	//mu = 0.001 + 0.999 * std::atan2(numerator, denominator);
	mu = 0.001 + 0.999 * std::atan2(numerator, denominator) + MathConstant::PI;
#else
	//mu = std::atan2(numerator, denominator);
	mu = std::atan2(numerator, denominator) + MathConstant::PI;
#endif
	assert(0.0 <= mu && mu < MathConstant::_2_PI);

	//
	numerator = 0.0;
	for (n = 0; n < N; ++n)
		numerator += gamma(n, state) * std::cos(observations(n, 0) - mu);

#if 0
	const double A = 0.001 + 0.999 * numerator / sumGamma;  // -1 < A < 1 (?).
#else
	const double A = numerator / sumGamma;  // -1 < A < 1 (?).
#endif
	// FIXME [modify] >> lower & upper bounds have to be adjusted.
	const double lb = -10000.0, ub = 10000.0;
	const std::size_t maxIteration = 100;
	const bool retval = one_dim_root_finding_using_f(A, lb, ub, maxIteration, kappas_[state]);
	assert(retval);

	// TODO [check] >>
	if (kappas_[state] < 0.0)  // kappa >= 0.0.
	{
		kappas_[state] = -kappas_[state];
		mu = std::fmod(mu + MathConstant::PI, MathConstant::_2_PI);
		assert(0.0 <= mu && mu < MathConstant::_2_PI);
	}

	// POSTCONDITIONS [] >>
	//	-. all mean directions have to be in [0, 2 * pi).
	//	-. all concentration parameters have to be greater than or equal to 0.
}

void VonMisesMixtureModel::doEstimateObservationDensityParametersByMAPUsingConjugatePrior(const size_t N, const unsigned int state, const dmatrix_type &observations, const dmatrix_type &gamma, const double sumGamma)
{
	// M-step.
	// reestimate observation(emission) distribution in each state.

	size_t n;

	double numerator = (*Rs_conj_)[state] * std::sin((*ms_conj_)[state]), denominator = (*Rs_conj_)[state] * std::cos((*ms_conj_)[state]);;
	for (n = 0; n < N; ++n)
	{
		numerator += gamma(n, state) * std::sin(observations(n, 0));
		denominator += gamma(n, state) * std::cos(observations(n, 0));
	}

	double &mu = mus_[state];

	// TODO [check] >> check the range of each mu, [0, 2 * pi).
#if 0
	//mu = 0.001 + 0.999 * std::atan2(numerator, denominator);
	mu = 0.001 + 0.999 * std::atan2(numerator, denominator) + MathConstant::PI;
#else
	//mu = std::atan2(numerator, denominator);
	mu = std::atan2(numerator, denominator) + MathConstant::PI;
#endif
	assert(0.0 <= mu && mu < MathConstant::_2_PI);

	//
	numerator = (*Rs_conj_)[state] * std::cos(mu - (*ms_conj_)[state]);
	denominator = sumGamma + (*cs_conj_)[state];
	for (n = 0; n < N; ++n)
		numerator += gamma(n, state) * std::cos(observations(n, 0) - mu);

#if 0
	const double A = 0.001 + 0.999 * numerator / denominator;  // -1 < A < 1 (?).
#else
	const double A = numerator / denominator;  // -1 < A < 1 (?).
#endif
	// FIXME [modify] >> lower & upper bounds have to be adjusted.
	const double lb = -10000.0, ub = 10000.0;
	const std::size_t maxIteration = 100;
	const bool retval = one_dim_root_finding_using_f(A, lb, ub, maxIteration, kappas_[state]);
	assert(retval);

	// TODO [check] >>
	if (kappas_[state] < 0.0)  // kappa >= 0.0.
	{
		kappas_[state] = -kappas_[state];
		mu = std::fmod(mu + MathConstant::PI, MathConstant::_2_PI);
		assert(0.0 <= mu && mu < MathConstant::_2_PI);
	}

	// POSTCONDITIONS [] >>
	//	-. all mean directions have to be in [0, 2 * pi).
	//	-. all concentration parameters have to be greater than or equal to 0.
}

void VonMisesMixtureModel::doEstimateObservationDensityParametersByMAPUsingEntropicPrior(const size_t N, const unsigned int state, const dmatrix_type &observations, const dmatrix_type &gamma, const double /*z*/, const bool /*doesTrimParameter*/, const bool isTrimmed, const double sumGamma)
{
	if (isTrimmed)
	{
		mus_[state] = 0.0;
		kappas_[state] = 0.0;
	}
	else
		doEstimateObservationDensityParametersByML(N, state, observations, gamma, sumGamma);
}

double VonMisesMixtureModel::doEvaluateMixtureComponentProbability(const unsigned int state, const dvector_type &observation) const
{
	// each observation are expressed as a random angle, 0 <= observation[0] < 2 * pi. [rad].
	//return 0.5 * std::exp(kappas_[k] * std::cos(observation[0] - mus_[state])) / (MathConstant::PI * boost::math::cyl_bessel_i(0.0, kappas_[state]));
	return evaluateVonMisesDistribution(observation[0], mus_[state], kappas_[state]);
}

void VonMisesMixtureModel::doGenerateObservationsSymbol(const unsigned int state, boost::numeric::ublas::matrix_row<dmatrix_type> &observation) const
{
	assert(!!targetDist_ && !!proposalDist_);

	targetDist_->setParameters(mus_[state], kappas_[state]);

#if 0
	// when using univariate normal proposal distribution.
	{
		// FIXME [modify] >> these parameters are incorrect.
		const double sigma = 1.55;
		const double k = 1.472;
		proposalDist_->setParameters(mus_[state], sigma, k);
	}
#else
	// when using univariate uniform proposal distribution.
	{
		const double lower = 0.0;
		const double upper = MathConstant::_2_PI;
		const UnivariateUniformProposalDistribution::vector_type mean_dir(1, mus_[state]);
		const double k = targetDist_->evaluate(mean_dir) * (upper - lower) * 1.05;
		proposalDist_->setParameters(lower, upper, k);
	}
#endif

	swl::RejectionSampling sampler(*targetDist_, *proposalDist_);

	swl::RejectionSampling::vector_type x(D_, 0.0);
	const std::size_t maxIteration = 1000;

	// the range of each observation, [0, 2 * pi)
	const bool retval = sampler.sample(x, maxIteration);
	assert(retval);
	observation[0] = x[0];
}

void VonMisesMixtureModel::doInitializeRandomSampleGeneration(const unsigned int seed /*= (unsigned int)-1*/) const
{
	if (!targetDist_) targetDist_.reset(new VonMisesTargetDistribution());
#if 0
	if (!proposalDist_) proposalDist_.reset(new UnivariateNormalProposalDistribution());
#else
	if (!proposalDist_) proposalDist_.reset(new UnivariateUniformProposalDistribution());
#endif

	if ((unsigned int)-1 != seed)
	{
		std::srand(seed);
		if (!!proposalDist_) proposalDist_->setSeed(seed);
	}
}

void VonMisesMixtureModel::doFinalizeRandomSampleGeneration() const
{
	targetDist_.reset();
	proposalDist_.reset();
}

bool VonMisesMixtureModel::doReadObservationDensity(std::istream &stream)
{
	if (1 != D_) return false;

	std::string dummy;
	stream >> dummy;
#if defined(__GNUC__)
	if (strcasecmp(dummy.c_str(), "von") != 0)
#elif defined(_MSC_VER)
	if (_stricmp(dummy.c_str(), "von") != 0)
#endif
		return false;

	stream >> dummy;
#if defined(__GNUC__)
	if (strcasecmp(dummy.c_str(), "Mises") != 0)
#elif defined(_MSC_VER)
	if (_stricmp(dummy.c_str(), "Mises") != 0)
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
	if (strcasecmp(dummy.c_str(), "kappa:") != 0)
#elif defined(_MSC_VER)
	if (_stricmp(dummy.c_str(), "kappa:") != 0)
#endif
		return false;

	// K
	for (size_t k = 0; k < K_; ++k)
		stream >> kappas_[k];

	return true;
}

bool VonMisesMixtureModel::doWriteObservationDensity(std::ostream &stream) const
{
	stream << "von Mises mixture:" << std::endl;

	// K
	stream << "mu:" << std::endl;
	for (size_t k = 0; k < K_; ++k)
		stream << mus_[k] << ' ';
	stream << std::endl;

	// K
	stream << "kappa:" << std::endl;
	for (size_t k = 0; k < K_; ++k)
		stream << kappas_[k] << ' ';
	stream << std::endl;

	return true;
}

void VonMisesMixtureModel::doInitializeObservationDensity(const std::vector<double> &lowerBoundsOfObservationDensity, const std::vector<double> &upperBoundsOfObservationDensity)
{
	// PRECONDITIONS [] >>
	//	-. std::srand() has to be called before this function is called.

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
			kappas_[k] = ((double)std::rand() / RAND_MAX) * (ub - lb) + lb;
		}
	}
	else if (numParameters == numLowerBound)
	{
		size_t k, idx = 0;
		for (k = 0; k < K_; ++k, ++idx)
			mus_[k] = ((double)std::rand() / RAND_MAX) * (upperBoundsOfObservationDensity[idx] - lowerBoundsOfObservationDensity[idx]) + lowerBoundsOfObservationDensity[idx];
		for (k = 0; k < K_; ++k, ++idx)
			kappas_[k] = ((double)std::rand() / RAND_MAX) * (upperBoundsOfObservationDensity[idx] - lowerBoundsOfObservationDensity[idx]) + lowerBoundsOfObservationDensity[idx];
	}

	for (size_t k = 0; k < K_; ++k)
	{
		// all concentration parameters have to be greater than or equal to 0.
		if (kappas_[k] < 0.0)
			kappas_[k] = -kappas_[k];
	}

#if defined(DEBUG) || defined(_DEBUG)
	for (size_t k = 0; k < K_; ++k)
	{
		assert(0.0 <= mus_[k] && mus_[k] < MathConstant::_2_PI);
		assert(kappas_[k] >= 0.0);
	}
#endif

	// POSTCONDITIONS [] >>
	//	-. all concentration parameters have to be greater than or equal to 0.
}

}  // namespace swl
