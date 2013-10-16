#include "swl/Config.h"
#include "swl/rnd_util/HmmWithVonMisesObservations.h"
#include "swl/math/MathConstant.h"
#include "RndUtilLocalApi.h"
#include "swl/rnd_util/RejectionSampling.h"
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/math/constants/constants.hpp>
#include <stdexcept>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

// [ref] swl/src/rnd_util/RndUtilLocalApi.cpp
bool one_dim_root_finding_using_f(const double A, const double lower, const double upper, const std::size_t maxIteration, double &kappa);
double evaluateVonMisesDistribution(const double x, const double mu, const double kappa);

HmmWithVonMisesObservations::HmmWithVonMisesObservations(const size_t K)
: base_type(K, 1), mus_(K, 0.0), kappas_(K, 0.0),  // 0-based index
  ms_conj_(), Rs_conj_(), cs_conj_(),
  targetDist_(), proposalDist_()
{
}

HmmWithVonMisesObservations::HmmWithVonMisesObservations(const size_t K, const dvector_type &pi, const dmatrix_type &A, const dvector_type &mus, const dvector_type &kappas)
: base_type(K, 1, pi, A), mus_(mus), kappas_(kappas),
  ms_conj_(), Rs_conj_(), cs_conj_(),
  targetDist_(), proposalDist_()
{
}

HmmWithVonMisesObservations::HmmWithVonMisesObservations(const size_t K, const dvector_type *pi_conj, const dmatrix_type *A_conj, const dvector_type *ms_conj, const dvector_type *Rs_conj, const dvector_type *cs_conj)
: base_type(K, 1, pi_conj, A_conj), mus_(K, 0.0), kappas_(K, 0.0),
  ms_conj_(ms_conj), Rs_conj_(Rs_conj), cs_conj_(cs_conj),
  targetDist_(), proposalDist_()
{
}

HmmWithVonMisesObservations::~HmmWithVonMisesObservations()
{
}

void HmmWithVonMisesObservations::doEstimateObservationDensityParametersByML(const size_t N, const unsigned int state, const dmatrix_type &observations, const dmatrix_type &gamma, const double denominatorA)
{
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
	denominator = denominatorA + gamma(N-1, state);
	numerator = 0.0;
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

void HmmWithVonMisesObservations::doEstimateObservationDensityParametersByML(const std::vector<size_t> &Ns, const unsigned int state, const std::vector<dmatrix_type> &observationSequences, const std::vector<dmatrix_type> &gammas, const size_t R, const double denominatorA)
{
	// reestimate observation(emission) distribution in each state.

	size_t n, r;
	double numerator = 0.0, denominator = 0.0;
	for (r = 0; r < R; ++r)
	{
		const dmatrix_type &observationr = observationSequences[r];
		const dmatrix_type &gammar = gammas[r];

		for (n = 0; n < Ns[r]; ++n)
		{
			numerator += gammar(n, state) * std::sin(observationr(n, 0));
			denominator += gammar(n, state) * std::cos(observationr(n, 0));
		}
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
	denominator = denominatorA;
	for (r = 0; r < R; ++r)
		denominator += gammas[r](Ns[r]-1, state);

	numerator = 0.0;
	for (r = 0; r < R; ++r)
	{
		const dmatrix_type &observationr = observationSequences[r];
		const dmatrix_type &gammar = gammas[r];

		for (n = 0; n < Ns[r]; ++n)
			numerator += gammar(n, state) * std::cos(observationr(n, 0) - mu);
	}

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

void HmmWithVonMisesObservations::doEstimateObservationDensityParametersByMAPUsingConjugatePrior(const size_t N, const unsigned int state, const dmatrix_type &observations, const dmatrix_type &gamma, const double denominatorA)
{
	// reestimate observation(emission) distribution in each state.

	size_t n;
	double numerator = (*Rs_conj_)(state) * std::sin((*ms_conj_)(state)), denominator = (*Rs_conj_)(state) * std::cos((*ms_conj_)(state));
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
	denominator = denominatorA + gamma(N-1, state) + (*cs_conj_)(state);
	numerator = (*Rs_conj_)(state) * std::cos(mu - (*ms_conj_)(state));
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

void HmmWithVonMisesObservations::doEstimateObservationDensityParametersByMAPUsingConjugatePrior(const std::vector<size_t> &Ns, const unsigned int state, const std::vector<dmatrix_type> &observationSequences, const std::vector<dmatrix_type> &gammas, const size_t R, const double denominatorA)
{
	// reestimate observation(emission) distribution in each state.

	size_t n, r;
	double numerator = (*Rs_conj_)(state) * std::sin((*ms_conj_)(state)), denominator = (*Rs_conj_)(state) * std::cos((*ms_conj_)(state));
	for (r = 0; r < R; ++r)
	{
		const dmatrix_type &observationr = observationSequences[r];
		const dmatrix_type &gammar = gammas[r];

		for (n = 0; n < Ns[r]; ++n)
		{
			numerator += gammar(n, state) * std::sin(observationr(n, 0));
			denominator += gammar(n, state) * std::cos(observationr(n, 0));
		}
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
	denominator = denominatorA + (*cs_conj_)(state);
	for (r = 0; r < R; ++r)
		denominator += gammas[r](Ns[r]-1, state);

	numerator = (*Rs_conj_)(state) * std::cos(mu - (*ms_conj_)(state));
	for (r = 0; r < R; ++r)
	{
		const dmatrix_type &observationr = observationSequences[r];
		const dmatrix_type &gammar = gammas[r];

		for (n = 0; n < Ns[r]; ++n)
			numerator += gammar(n, state) * std::cos(observationr(n, 0) - mu);
	}

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

void HmmWithVonMisesObservations::doEstimateObservationDensityParametersByMAPUsingEntropicPrior(const size_t N, const unsigned int state, const dmatrix_type &observations, const dmatrix_type &gamma, const double /*z*/, const double /*terminationTolerance*/, const size_t /*maxIteration*/, const double denominatorA)
{
	doEstimateObservationDensityParametersByML(N, state, observations, gamma, denominatorA);
}

void HmmWithVonMisesObservations::doEstimateObservationDensityParametersByMAPUsingEntropicPrior(const std::vector<size_t> &Ns, const unsigned int state, const std::vector<dmatrix_type> &observationSequences, const std::vector<dmatrix_type> &gammas, const double /*z*/, const size_t R, const double /*terminationTolerance*/, const size_t /*maxIteration*/, const double denominatorA)
{
	doEstimateObservationDensityParametersByML(Ns, state, observationSequences, gammas, R, denominatorA);
}

double HmmWithVonMisesObservations::doEvaluateEmissionProbability(const unsigned int state, const boost::numeric::ublas::matrix_row<const dmatrix_type> &observation) const
{
	// each observation are expressed as a random angle, 0 <= observation[0] < 2 * pi. [rad].
	//return 0.5 * std::exp(kappas_[state] * std::cos(observation[0] - mus_[state])) / (MathConstant::PI * boost::math::cyl_bessel_i(0.0, kappas_[state]));
	return evaluateVonMisesDistribution(observation[0], mus_[state], kappas_[state]);
}

void HmmWithVonMisesObservations::doGenerateObservationsSymbol(const unsigned int state, boost::numeric::ublas::matrix_row<dmatrix_type> &observation, const unsigned int seed /*= (unsigned int)-1*/) const
{
	// PRECONDITIONS [] >>
	//	-. std::srand() had to be called before this function is called.

	if (!targetDist_) targetDist_.reset(new VonMisesTargetDistribution());
	targetDist_->setParameters(mus_[state], kappas_[state]);

#if 0
	if (!proposalDist_) proposalDist_.reset(new UnivariateNormalProposalDistribution());

	{
		// FIXME [modify] >> these parameters are incorrect.
		const double sigma = 1.55;
		const double k = 1.472;
		proposalDist_->setParameters(mus_[state], sigma, k);
	}
#else
	if (!proposalDist_) proposalDist_.reset(new UnivariateUniformProposalDistribution());

	{
		const double lower = 0.0;
		const double upper = MathConstant::_2_PI;
		const UnivariateUniformProposalDistribution::vector_type mean_dir(1, mus_[state]);
		const double k = targetDist_->evaluate(mean_dir) * (upper - lower) * 1.05;
		proposalDist_->setParameters(lower, upper, k);
	}
#endif

	if ((unsigned int)-1 != seed)
		proposalDist_->setSeed(seed);

	swl::RejectionSampling sampler(*targetDist_, *proposalDist_);

	swl::RejectionSampling::vector_type x(D_, 0.0);
	const std::size_t maxIteration = 1000;

	// the range of each observation, [0, 2 * pi)
	const bool retval = sampler.sample(x, maxIteration);
	assert(retval);
	observation[0] = x[0];
}

bool HmmWithVonMisesObservations::doReadObservationDensity(std::istream &stream)
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
	if (strcasecmp(dummy.c_str(), "Mises:") != 0)
#elif defined(_MSC_VER)
	if (_stricmp(dummy.c_str(), "Mises:") != 0)
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

bool HmmWithVonMisesObservations::doWriteObservationDensity(std::ostream &stream) const
{
	stream << "von Mises:" << std::endl;

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

void HmmWithVonMisesObservations::doInitializeObservationDensity(const std::vector<double> &lowerBoundsOfObservationDensity, const std::vector<double> &upperBoundsOfObservationDensity)
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
