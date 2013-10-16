#include "swl/Config.h"
#include "swl/rnd_util/HMM.h"
#include "swl/math/MathUtil.h"
#include <gsl/gsl_sf_lambert.h>
#include <gsl/gsl_linalg.h>
#include <boost/math/constants/constants.hpp>
#include <algorithm>
#include <numeric>
#include <limits>
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

bool HMM::computeMAPEstimateOfMultinomialUsingEntropicPrior(const std::vector<double> &omega, const double &z, std::vector<double> &theta, double &logLikelihood, const double terminationTolerance, const std::size_t maxIteration, const bool doesInitializeLambdaFirst /*= true*/) const
{
	// [ref] "Structure Learning in Conditional Probability Models via an Entropic Prior and Parameter Extinction", M. Brand, Neural Computation, 1999.

#if 0
	// FIXME [fix] >> not working.

	const std::size_t K = omega.size();
	//theta.resize(K, 0.0);
	std::fill(theta.begin(), theta.end(), 0.0);
	std::vector<bool> extinctionFlag(K, false);
	std::vector<double> omega_pos(K, 0.0);

	// ignore negative evidence values.
	std::size_t k;
	for (k = 0; k < K; ++k)
		omega_pos[k] = std::max(0.0, omega[k]);

	double W = 0.0;
	double sumTheta = 0.0;

	const double _1_e = -1.0 / boost::math::constants::e<double>();
	const double eps = 1.0e-20;

	std::vector<double> prevTheta(K, -std::numeric_limits<double>::max());

	// METHOD #1: lambda -> theta -> lambda -> ...
	if (doesInitializeLambdaFirst)
	{
		// initialize lambda.
		double sumOmega = 0.0, ElogOmega = 0.0;
		for (std::vector<double>::const_iterator cit = omega_pos.begin(); cit != omega_pos.end(); ++cit)
		{
			sumOmega += *cit;
			ElogOmega += std::log(*cit);
		}
		ElogOmega /= double(K);
		std::vector<double> lambda(K, -sumOmega - ElogOmega);

		double x;
		std::size_t iteration = 0;
		bool looping = true;
		while (iteration < maxIteration)
		{
			// calculate theta given lambda.
			for (k = 0; k < K; ++k)
			{
				x = -omega_pos[k] * std::exp(1.0 + lambda[k]);
				if (x < _1_e)
				{
					assert(x >= _1_e);
					return false;
				}

#if 1
				// the principal branch of the Lambert W function, W_0(x).
				//	W >= -1 for x >= -1/e.
				gsl_sf_result W0e;
				if (0 != gsl_sf_lambert_W0_e(x, &W0e)) return false;
				W = W0e.val;
#else
				// the secondary real-valued branch of the LambertW function, W_{−1}(x).
				//	W <= -1 for -1/e <= x < 0.
				//	W = W_0 for x >= 0.
				gsl_sf_result Wm1e;
				if (0 != gsl_sf_lambert_Wm1_e(x, &Wm1e)) return false;
				W = Wm1e.val;
#endif

				// TODO [check] >> when W = 0, omega(k) = 0 ?
				theta[k] = std::fabs(W) >= eps ? (-omega_pos[k] / W) : 0.0;
			}

			// normalize theta.
			sumTheta = std::accumulate(theta.begin(), theta.end(), 0.0);
			assert(std::fabs(sumTheta) >= eps);
			for (k = 0; k < K; ++k)
			{
				theta[k] /= sumTheta;

				// FIXME [check] >>
				//assert(theta[k] > 0.0);
				if (theta[k] < eps) extinctionFlag[k] = true;
			}

			//
			looping = false;
			for (k = 0; k < K; ++k)
				if (std::fabs(theta[k] - prevTheta[k]) >= terminationTolerance)
				{
					looping = true;
					break;
				}
			if (!looping) break;

			// calculate lambda given theta.
			for (k = 0; k < K; ++k)
				// FIXME [check] >>
				lambda[k] = extinctionFlag[k] ? 0.0 : ((-omega_pos[k] / theta[k]) - std::log(theta[k]) - 1.0);

			prevTheta.assign(theta.begin(), theta.end());
			++iteration;
		}
	}
	// METHOD #2: theta -> lambda -> theta -> ...
	else
	{
		// initialize theta.
		{
			const double sumOmega = std::accumulate(omega_pos.begin(), omega_pos.end(), 0.0);
			assert(std::fabs(sumOmega) >= eps);
			const double val = 1.0 - 1.0 / sumOmega;

			for (k = 0; k < K; ++k)
				theta[k] = std::pow(omega_pos[k], val);

			// normalize theta.
			sumTheta = std::accumulate(theta.begin(), theta.end(), 0.0);
			assert(std::fabs(sumTheta) >= eps);
			for (k = 0; k < K; ++k)
			{
				theta[k] /= sumTheta;

				// FIXME [check] >>
				//assert(theta[k] > 0.0);
				if (theta[k] < eps) extinctionFlag[k] = true;
			}
		}

		std::vector<double> lambda(K, 0.0);

		double x;
		std::size_t iteration = 0;
		bool looping = true;
		while (iteration < maxIteration)
		{
			// calculate lambda given theta.
			for (k = 0; k < K; ++k)
				// FIXME [check] >>
				lambda[k] = extinctionFlag[k] ? 0.0 : ((-omega_pos[k] / theta[k]) - std::log(theta[k]) - 1.0);

			// calculate theta given lambda.
			for (k = 0; k < K; ++k)
			{
				x = -omega_pos[k] * std::exp(1.0 + lambda[k]);
				if (x < _1_e)
				{
					assert(x >= _1_e);
					return false;
				}

#if 1
				// the principal branch of the Lambert W function, W_0(x).
				//	W >= -1 for x >= -1/e.
				gsl_sf_result W0e;
				if (0 != gsl_sf_lambert_W0_e(x, &W0e)) return false;
				W = W0e.val;
#else
				// the secondary real-valued branch of the LambertW function, W_{−1}(x).
				//	W <= -1 for -1/e <= x < 0.
				//	W = W_0 for x >= 0.
				gsl_sf_result Wm1e;
				if (0 != gsl_sf_lambert_Wm1_e(x, &Wm1e)) return false;
				W = Wm1e.val;
#endif
				
				// TODO [check] >> when W = 0, omega(k) = 0 ?
				theta[k] = std::fabs(W) >= eps ? (-omega_pos[k] / W) : 0.0;
			}

			// normalize theta.
			sumTheta = std::accumulate(theta.begin(), theta.end(), 0.0);
			assert(std::fabs(sumTheta) >= eps);
			for (k = 0; k < K; ++k)
			{
				theta[k] /= sumTheta;

				// FIXME [check] >>
				//assert(theta[k] > 0.0);
				if (theta[k] < eps) extinctionFlag[k] = true;
			}

			//
			looping = false;
			for (k = 0; k < K; ++k)
				if (std::fabs(theta[k] - prevTheta[k]) >= terminationTolerance)
				{
					looping = true;
					break;
				}
			if (!looping) break;

			prevTheta.assign(theta.begin(), theta.end());
			++iteration;
		}
	}
#else
	// warning: this is not numerically stable for sum(omega) < 1 or large z > 0.

	// theta: multinomial parameters.
	// omega: evidence vector (default: 1).
	// z:     exponent on prior (default: 1).

	// [ref] https://meteo.unican.es/trac/MLToolbox/browser/MLToolbox/MeteoLab/BayesNets/BNT/Entropic/entropic_map_estimate.m?rev=1.
	// The entropic prior says P(theta) \propto exp(-H(theta)), where H(.) is the entropy.
	//
	// z = 1 (default) is min. entropy.
	// z = 0 is max. likelihood.
	// z = -1 is max. entropy.
	// z = -inf corresponds to very high temperature (good for initialization).
	//
	// Based on "Structure learning in conditional probability models via an entropic prior and parameter extinction", M. Brand, Neural Computation, 1999.
	//
	// For the z ~= 1 case, see "Pattern discovery via entropy minimization", M. Brand, AI & Statistics, 1999. Equation numbers refer to this paper.

	extern double loglambertw0(double y);
	extern double loglambertwn1(double y);

	const std::size_t KK = omega.size();

	// Trivial case: only one parameter.
	if (1 == KK)
	{
		//std::fill(theta.begin(), theta.end(), 1.0);
		theta[0] = 1.0;
		logLikelihood = 1.0;
		return true;
	}

	std::size_t k;

	// Special case: heat death.
	if (z >= std::numeric_limits<double>::max())
	{
		// construct uniform distribution.

		std::size_t count = 0;
		for (k = 0; k < KK; ++k)
		{
			if (omega[k] >= 0.0)
				++count;
		}

		if (count > 0)
		{
			const double prob = 1.0 / (double)count;
			for (k = 0; k < KK; ++k)
				theta[k] = omega[k] >= 0.0 ? prob : 0.0;
		}
		else
			std::fill(theta.begin(), theta.end(), 1.0 / KK);

		logLikelihood = z * std::log((double)count);  // posterior dominated by prior.
		return true;
	}

	const double eps = 1.0e-20;

	//theta.resize(KK, 0.0);
	std::fill(theta.begin(), theta.end(), 0.0);
	logLikelihood = 0.0;

	{
		std::vector<double> omega_valid;
		std::vector<std::size_t> valid_omega_idx;
		omega_valid.reserve(KK);
		valid_omega_idx.reserve(KK);

		// check for values in omega that are very close to zero, and ignore negative evidence values.
		for (k = 0; k < KK; ++k)
		{
			if (omega[k] >= eps)
			{
				valid_omega_idx.push_back(k);
				omega_valid.push_back(omega[k]);
			}
		}

		const std::size_t KK_valid = omega_valid.size();
		if (KK_valid != KK)
		{
			//std::cerr << "At least one omega is very close to zero or less than zero" << std::endl;

			if (KK_valid > 1)
			{
				// Two or more nonzero parameters. Skip those that are zero.

				std::vector<double> theta_pos(KK_valid, 0.0);
				if (computeMAPEstimateOfMultinomialUsingEntropicPrior(omega_valid, z, theta_pos, logLikelihood, terminationTolerance, maxIteration, doesInitializeLambdaFirst))
				{
					for (k = 0; k < KK_valid; ++k)
						theta[valid_omega_idx[k]] = theta_pos[k];
					return true;
				}
				else
				{
					logLikelihood = 0.0;
					return false;
				}
			}
			else if (1 == KK_valid)
			{
				// Only one nonzero parameter. Return spike distribution.

				theta[valid_omega_idx[0]] = 1.0;
				logLikelihood = 0.0;

				// TODO [check] >> does we have to return true?
				return true;
			}
			else
			{
				// Everything is zero. Return 0s.

				//std::fill(theta.begin(), theta.end(), 0.0);
				logLikelihood = 0.0;

				// TODO [check] >> does we have to return true?
				return true;
			}
		}
	}

	// Fixpoint loop.
	const double sumOmega = std::accumulate(omega.begin(), omega.end(), 0.0);

	logLikelihood = -std::numeric_limits<double>::max();

	std::vector<double> logOmega(KK, 0.0), logTheta(KK, 0.0), omegaZ(KK, 0.0), logOmegaZ_pkz(KK, 0.0);
	for (k = 0; k < KK; ++k)
	{
		logOmega[k] = std::log(omega[k]);

		theta[k] = (omega[k] / sumOmega) + std::numeric_limits<double>::min();
		logTheta[k] = std::log(theta[k]);
	}

	double minLambda, maxLambda, lambda;
	const bool zIsZero = std::fabs(z) < eps;
	if (zIsZero)
	{
		minLambda = 0.0;
		maxLambda = std::numeric_limits<double>::max();
		lambda = sumOmega;
	}
	else
	{
		const double logZ = std::log(std::fabs(z));
		for (k = 0; k < KK; ++k)
		{
			omegaZ[k] = omega[k] / z;
			logOmegaZ_pkz[k] = logOmega[k] - logZ;
		}

		if (z < 0.0)
		{
			minLambda = *std::max_element(logOmegaZ_pkz.begin(), logOmegaZ_pkz.end()) - 700.0;
		}
		else
		{
			// For z > 0, we need to restrict minLambda so that the argument to loglambertwn1() below stays within range.
			minLambda = *std::max_element(logOmegaZ_pkz.begin(), logOmegaZ_pkz.end()) + 2.0;
			minLambda *= 1.0 + eps * (double)MathUtil::sign(minLambda);
		}
		maxLambda = *std::min_element(logOmega.begin(), logOmega.end()) + 700.0;
		lambda = std::accumulate(omegaZ.begin(), omegaZ.end(), 0.0) + 1.0 + *std::max_element(logTheta.begin(), logTheta.end());
	}
	lambda = std::min(maxLambda, std::max(minLambda, lambda));

	double oldLambda = lambda;
	double dLambda = std::numeric_limits<double>::max();
	double oldLogLikelihood = 0.0, oldDLogLikelihood = 0.0, oldDLambda = 0.0;
	double sumTheta = 0.0;
	std::vector<double> oldTheta(KK, 0.0);

	const int signZ = MathUtil::sign(z);
	const double PHI = (std::sqrt(5.0) - 1.0) * 0.5;
	const double NPHI = 1.0 - PHI;

	// Iterate fixpoint untila numerical error intrudes.
	if (minLambda < maxLambda)
	{
		double dLogLikelihood = std::numeric_limits<double>::max();
		std::size_t iteration = 0;
		while (iteration < maxIteration)
		{
			// Store previous values.
			oldTheta.assign(theta.begin(), theta.end());
			oldLogLikelihood = logLikelihood;
			oldDLogLikelihood = dLogLikelihood;
			oldDLambda = dLambda;

			// Step theta (inverse fixpoint).
			if (zIsZero)
			{
				for (k = 0; k < KK; ++k)
					theta[k] = std::max(omega[k] / lambda, 0.0);
			}
			else if (z < 0.0)
			{
				for (k = 0; k < KK; ++k)
					theta[k] = std::max(omega[k] / loglambertw0(lambda - 1.0 - logOmegaZ_pkz[k]), 0.0);
			}
			else
			{
				for (k = 0; k < KK; ++k)
					theta[k] = std::max(omega[k] / loglambertwn1(lambda - 1.0 - logOmegaZ_pkz[k]), 0.0);
			}

			sumTheta = std::accumulate(theta.begin(), theta.end(), 0.0);
			assert(std::fabs(sumTheta) >= eps);
			for (k = 0; k < KK; ++k)
			{
				theta[k] = (theta[k] / sumTheta) + std::numeric_limits<double>::min();
				logTheta[k] = std::log(theta[k]);
			}

			// Compute new entropic MLE log-likelihood.
			logLikelihood = 0.0;
			for (k = 0; k < KK; ++k)
				logLikelihood += (omega[k] + z * theta[k]) * logTheta[k];
			dLogLikelihood = logLikelihood - oldLogLikelihood;

			// Compare and save.
			if (std::fabs(dLogLikelihood) < eps)
			{
				if (signZ != MathUtil::sign(dLambda))
					break;

				// Back up half a step.
				theta.assign(oldTheta.begin(), oldTheta.end());
				logLikelihood = oldLogLikelihood;
				dLambda *= 0.5;
				lambda -= dLambda;
			}
			else if (dLogLikelihood < 0.0)
			{
				// Golden mean.
				theta.assign(oldTheta.begin(), oldTheta.end());
				if (oldDLogLikelihood + dLogLikelihood <= 0.0)
				{
					logLikelihood = oldLogLikelihood;
					for (k = 0; k < KK; ++k)
						logTheta[k] = std::log(theta[k]);

					break;
				}

				logLikelihood = oldLogLikelihood;
				lambda = NPHI * lambda + PHI * oldLambda;
				dLambda = lambda - oldLambda;
				oldDLambda = std::numeric_limits<double>::max();
			}
			else
			{
				// Improvement.
				oldLambda = lambda;
				if (zIsZero)
				{
					double sum = 0.0;
					for (k = 0; k < KK; ++k)
						sum += omega[k] / theta[k];
					lambda = sum / KK;
				}
				else
				{
					double sum = 0.0;
					for (k = 0; k < KK; ++k)
					{
						sum += (omegaZ[k] / theta[k]) + logTheta[k];
					}
					lambda = 1.0 + sum / KK;
				}
				lambda = std::min(maxLambda, std::max(minLambda, lambda));
				dLambda = lambda - oldLambda;
			}

			++iteration;
		}
	}
	else
	{
		// The range of logomegaz_pkz seems totally out of whack -- what the heck, let's just skip the fixpoint loop.
		for (k = 0; k < KK; ++k)
			logTheta[k] = std::log(theta[k]);
	}

	// Very close now; polish up with 2nd order Newton-Raphson with bisection.
	const double nm1 = (double)KK - 1.0;  // n - 1.
	//const double nonm1 = 1.0 + 1.0 / nm1;  // n / (n - 1).

	logLikelihood = 0.0;
	for (k = 0; k < KK; ++k)
		logLikelihood += (omega[k] + z * theta[k]) * logTheta[k];

	// turns off the error handler by defining an error handler which does nothing.
	gsl_error_handler_t *old_error_handler = gsl_set_error_handler_off();

	std::vector<double> ratio(KK, 0.0), delta(KK, 0.0), delta_theta(KK, 0.0), dTheta(KK, 0.0), ddTheta(KK, 0.0), dddTheta(KK, 0.0), factor(KK, 0.0);
	gsl_matrix *jacobian = gsl_matrix_alloc(KK, KK);
	gsl_permutation *p = gsl_permutation_alloc(KK);
	gsl_vector *b = gsl_vector_alloc(KK), *x = gsl_vector_alloc(KK);
	int signum;
	double maxDeltaTheta;
	std::size_t i;
	std::size_t iteration = 0;
	while (iteration < maxIteration)
	{
		// Store previous values.
		oldLogLikelihood = logLikelihood;
		oldTheta.assign(theta.begin(), theta.end());

		// Function we want root of.
		for (k = 0; k < KK; ++k)
		{
			ratio[k] = omega[k] / theta[k];
			dTheta[k] = ratio[k] + z * logTheta[k];
			ddTheta[k] = (z - ratio[k]) / theta[k];  // 1st derivative of dTheta.
		}
		const double meanDTheta = std::accumulate(dTheta.begin(), dTheta.end(), 0.0) / (double)KK;
		for (k = 0; k < KK; ++k)
			dTheta[k] -= meanDTheta;

		// 1st order Newton-Raphson via Jacobian.
		for (k = 0; k < KK; ++k)
		{
			for (i = 0; i < KK; ++i)
				gsl_matrix_set(jacobian, k, i, i == k ? ddTheta[i] : (-ddTheta[i] / nm1));
			gsl_vector_set(b, k, dTheta[k]);
		}

		// LU decomposition.
		const int status1 = gsl_linalg_LU_decomp(jacobian, p, &signum);
		const int status2 = gsl_linalg_LU_solve(jacobian, p, b, x);
		if (!status1 && !status2)
		{
			for (k = 0; k < KK; ++k)
				delta[k] = gsl_vector_get(x, k);
		}
		else
		{
			bool isAllFactorPositive = true;
			for (k = 0; k < KK; ++k)
			{
				dddTheta[k] = (2.0 * ratio[k] - z) / (theta[k] * theta[k]);  // 2nd derivative of dTheta.
				factor[k] = ddTheta[k] * ddTheta[k] - 2.0 * dTheta[k] * dddTheta[k];
				if (factor[k] < 0.0) isAllFactorPositive = false;
			}

			if (isAllFactorPositive)
			{
				for (k = 0; k < KK; ++k)
					delta[k] = (ddTheta[k] + std::sqrt(factor[k])) / dddTheta[k];  // 2nd order Newton-Raphson.

				//
				bool deltaFlag = true;
				double sumAbsDelta = 0.0;
				for (k = 0; k < KK; ++k)
				{
					deltaFlag &= delta[k] > theta[k];
					sumAbsDelta += std::fabs(delta[k]);
				}
				if (deltaFlag || sumAbsDelta < eps)
				{
					for (k = 0; k < KK; ++k)
						delta[k] = (ddTheta[k] - std::sqrt(factor[k])) / dddTheta[k];  // 2nd order Newton-Raphson.
				}

				//
				deltaFlag = true;
				sumAbsDelta = 0.0;
				for (k = 0; k < KK; ++k)
				{
					deltaFlag &= delta[k] > theta[k];
					sumAbsDelta += std::fabs(delta[k]);
				}
				if (deltaFlag || sumAbsDelta < eps)
				{
					for (k = 0; k < KK; ++k)
					{
						if (std::fabs(ddTheta[k]) < eps) ddTheta[k] = std::numeric_limits<double>::max();
						delta[k] = dTheta[k] / ddTheta[k];  // 1st order Newton-Raphson.
					}
				}
			}
			else
			{
				for (k = 0; k < KK; ++k)
				{
					if (std::fabs(ddTheta[k]) < eps) ddTheta[k] = std::numeric_limits<double>::max();
					delta[k] = dTheta[k] / ddTheta[k];  // 1st order Newton-Raphson.
				}
			}
		}

		// If (omitted) higher-order terms are significant, must scale down delta.
		for (k = 0; k < KK; ++k)
			delta_theta[k] = delta[k] / theta[k];
		maxDeltaTheta = *std::max_element(delta_theta.begin(), delta_theta.end());
		if (maxDeltaTheta > 1.0)
		{
			const double val = NPHI / maxDeltaTheta;
			for (k = 0; k < KK; ++k)
				delta[k] *= val;
		}
		for (k = 0; k < KK; ++k)
			theta[k] = std::max(theta[k] - delta[k], 0.0);
		sumTheta = std::accumulate(theta.begin(), theta.end(), 0.0);
		assert(std::fabs(sumTheta) >= eps);
		for (k = 0; k < KK; ++k)
		{
			theta[k] = (theta[k] / sumTheta) + std::numeric_limits<double>::min();
			logTheta[k] = std::log(theta[k]);
		}

		logLikelihood = 0.0;
		for (k = 0; k < KK; ++k)
			logLikelihood += (omega[k] + z * theta[k]) * logTheta[k];

		if (logLikelihood <= oldLogLikelihood)
		{
			for (k = 0; k < KK; ++k)
			{
				theta[k] = theta[k] * NPHI + oldTheta[k] * PHI;
				logTheta[k] = std::log(theta[k]);
			}

			logLikelihood = 0.0;
			for (k = 0; k < KK; ++k)
				logLikelihood += (omega[k] + z * theta[k]) * logTheta[k];

			if (logLikelihood <= oldLogLikelihood)
			{
				theta = oldTheta;
				logLikelihood = oldLogLikelihood;
				break;
			}
		}

		++iteration;
	}

	gsl_permutation_free(p);
	gsl_matrix_free(jacobian);
	gsl_vector_free(b);
	gsl_vector_free(x);

	// restore original error handler.
	gsl_set_error_handler(old_error_handler);
#endif

	if (iteration >= maxIteration)
		std::cerr << "max. iteration exceeded: " << iteration << std::endl;

	return true;
}

}  // namespace swl
