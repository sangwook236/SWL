#include "swl/Config.h"
#include "RndUtilLocalApi.h"
#include "swl/math/MathUtil.h"
#include <gsl/gsl_sf_lambert.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_errno.h>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/blas.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/constants/constants.hpp>
#include <algorithm>
#include <numeric>
#include <limits>
#include <ctime>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


extern double loglambertw0(double y);
extern double loglambertwn1(double y);

namespace swl {

//--------------------------------------------------------------------------
//

double determinant_by_lu(const boost::numeric::ublas::matrix<double> &m)
{
	// create a working copy of the m
	boost::numeric::ublas::matrix<double> A(m);
    boost::numeric::ublas::permutation_matrix<std::size_t> pm(A.size1());
    if (boost::numeric::ublas::lu_factorize(A, pm))
        return 0.0;
	else
	{
	    double det = 1.0;
		for (std::size_t i = 0; i < pm.size(); ++i)
			det *= (pm(i) == i) ? A(i, i) : -A(i, i);

		return det;
    }
}

bool inverse_by_lu(const boost::numeric::ublas::matrix<double> &m, boost::numeric::ublas::matrix<double> &inv)
{
	// create a working copy of the m
	boost::numeric::ublas::matrix<double> A(m);
	// create a permutation matrix for the LU factorization
	boost::numeric::ublas::permutation_matrix<std::size_t> pm(A.size1());

	// perform LU factorization
	if (boost::numeric::ublas::lu_factorize(A, pm))
		return false;
	else
	{
		// create identity matrix of inv
		inv.assign(boost::numeric::ublas::identity_matrix<double>(A.size1()));

		// back-substitute to get the inverse
		boost::numeric::ublas::lu_substitute(A, pm, inv);

		return true;
	}
}

double det_and_inv_by_lu(const boost::numeric::ublas::matrix<double> &m, boost::numeric::ublas::matrix<double> &inv)
{
	// create a working copy of the m
	boost::numeric::ublas::matrix<double> A(m);
	// create a permutation matrix for the LU factorization
	boost::numeric::ublas::permutation_matrix<std::size_t> pm(A.size1());

	// perform LU factorization
	if (boost::numeric::ublas::lu_factorize(A, pm))
		return 0.0;
	else
	{
		// create identity matrix of inv
		inv.assign(boost::numeric::ublas::identity_matrix<double>(A.size1()));

		// back-substitute to get the inverse
		boost::numeric::ublas::lu_substitute(A, pm, inv);

		//
	    double det = 1.0;
		for (std::size_t i = 0; i < pm.size(); ++i)
			det *= (pm(i) == i) ? A(i, i) : -A(i, i);

		return det;
	}
}

//--------------------------------------------------------------------------
//

double kappa_objective_function(double x, void *params)
{
	try
	{
		const double *A = (double *)params;

		// TODO [check] >>
		if (-700.0 <= x && x <= 700.0)
			return boost::math::cyl_bessel_i(1.0, x) / boost::math::cyl_bessel_i(0.0, x) - *A;
		else
		{
			// [ref] "Directional Statistics" by K. Mardia, pp. 40
/*
			// for small kappa
			//return 0.5 * x * (1.0 - x*x/8.0 + x*x*x*x/48.0 - x*x*x*x*x*x*11.0/3072.0) - *A;
			const double x2 = x * x;
			//return 0.5 * x * (1.0 + x2 * (-1.0/8.0 + x2 * (1.0/48.0 - x2 * 11.0/3072.0))) - *A;
			return 0.5 * x * (1.0 + x2 * (-0.125 + x2 * (0.020833333333333 - x2 * 0.003580729166667))) - *A;
*/
			// for large kappa
			//const double x2 = x * x;
			//const double val = x >= 0 ? (1.0 - 1.0 / (2.0 * x) - 1.0 / (8.0 * x2) - 1.0 / (8.0 * x2 * x)) : (-1.0 - 1.0 / (2.0 * x) + 1.0 / (8.0 * x2) - 1.0 / (8.0 * x2 * x));
			const double val = x >= 0 ? (1.0 - (0.5 + (0.125 + 0.125 / x) / x) / x) : (-1.0 - (0.5 - (0.125 - 0.125 / x) / x) / x);
			return val - *A;
		}
	}
	catch (const std::exception &)
	{
		assert(false);
		return 0.0;
	}
}

bool one_dim_root_finding_using_f(const double A, const double lower, const double upper, const std::size_t maxIteration, double &kappa)
{
	gsl_function func;
	func.function = &kappa_objective_function;
	func.params = (void *)&A;

	//const gsl_root_fsolver_type *T = gsl_root_fsolver_bisection;
	//const gsl_root_fsolver_type *T = gsl_root_fsolver_falsepos;
	const gsl_root_fsolver_type *T = gsl_root_fsolver_brent;
	gsl_root_fsolver *s = gsl_root_fsolver_alloc(T);

	double x_lo = lower, x_hi = upper;
	gsl_root_fsolver_set(s, &func, x_lo, x_hi);

	//std::cout << "===== using " << gsl_root_fsolver_name(s) << " method =====" << std::endl;
	//std::cout << std::setw(5) << "iter" << " [" << std::setw(9) << "lower" << ", " << std::setw(9) << "upper" << "] " << std::setw(9) << "root" << std::setw(10) << "err(est)" << std::endl;

	int status;
	std::size_t iter = 0;
	kappa = 0.0;
	do
	{
		++iter;

		status = gsl_root_fsolver_iterate(s);
		kappa = gsl_root_fsolver_root(s);
		x_lo = gsl_root_fsolver_x_lower(s);
		x_hi = gsl_root_fsolver_x_upper(s);
		status = gsl_root_test_interval(x_lo, x_hi, 0, 0.001);

		if (GSL_SUCCESS == status)
		{
			//std::cout << "converged" << std::endl;
			return true;
		}

		//std::cout << std::setw(5) << iter << " [" << std::setw(9) << x_lo << ", " << std::setw(9) << x_hi << "] " << std::setw(9) << kappa << std::setw(10) << (x_hi - x_lo) << std::endl;
	} while (GSL_CONTINUE == status && iter < maxIteration);

	if (GSL_SUCCESS != status)
	{
		//std::cout << "not converged" << std::endl;
		kappa = 0.0;
		return false;
	}
	else return true;
}

//--------------------------------------------------------------------------
// von Mises distribution

double evaluateVonMisesDistribution(const double x, const double mu, const double kappa)
{
	try
	{
		// TODO [check] >>
		if (-700.0 <= kappa && kappa <= 700.0)
			return 0.5 * std::exp(kappa * std::cos(x - mu)) / (MathConstant::PI * boost::math::cyl_bessel_i(0.0, kappa));
		else
		{
			// [ref] "Directional Statistics" by K. Mardia, pp. 40
			const double m = 16.0;  // 4 * p^2
			//const double Ip = std::exp(kappa) * (1.0 - (m - 1.0) / (8.0 * kappa) + (m - 1.0) * (m - 9.0) / (2.0 * 64.0 * kappa * kappa) - (m - 1.0) * (m - 9.0) * (m - 25.0) / (6.0 * 512.0 * kappa * kappa * kappa)) / std::sqrt(MathConstant::_2_PI * kappa);
			const double Ip = std::exp(kappa) * (1.0 - (0.125 * (m - 1.0) / kappa) * (1.0 - (0.0625 * (m - 9.0) / kappa) * (1.0 - (m - 25.0) / (24.0 * kappa)))) / std::sqrt(MathConstant::_2_PI * kappa);
			return 0.5 * std::exp(kappa * std::cos(x - mu)) / (MathConstant::PI * Ip);
		}
	}
	catch (const std::exception &)
	{
		assert(false);
		return 0.0;
	}
}

//--------------------------------------------------------------------------
// von Mises target distribution

double VonMisesTargetDistribution::evaluate(const vector_type &x) const
{
	//return 0.5 * std::exp(kappa_ * std::cos(x[0] - mean_direction_)) / (MathConstant::PI * boost::math::cyl_bessel_i(0.0, kappa_));
	return evaluateVonMisesDistribution(x[0], mean_direction_, kappa_);
}

//--------------------------------------------------------------------------
// univariate normal proposal distribution

UnivariateNormalProposalDistribution::UnivariateNormalProposalDistribution()
: base_type(), mean_(0.0), sigma_(1.0), baseGenerator_(), generator_(baseGenerator_, boost::normal_distribution<>(mean_, sigma_))
{}

double UnivariateNormalProposalDistribution::evaluate(const vector_type &x) const
{
	boost::math::normal dist(mean_, sigma_);
	return k_ * boost::math::pdf(dist, x[0]);
}

void UnivariateNormalProposalDistribution::sample(vector_type &sample) const
{
	// 0 <= x < 2 * pi
	sample[0] = swl::MathUtil::wrap(generator_(), 0.0, MathConstant::_2_PI);
}

void UnivariateNormalProposalDistribution::setParameters(const double mean, const double sigma, const double k /*= 1.0*/)
{
	mean_ = mean;
	sigma_ = sigma;
	k_ = k;

	generator_.distribution().param(boost::normal_distribution<>::param_type(mean_, sigma_));
}

void UnivariateNormalProposalDistribution::setSeed(const unsigned int seed)
{
	baseGenerator_.seed(seed);
}

//--------------------------------------------------------------------------
// univariate uniform proposal distribution

UnivariateUniformProposalDistribution::UnivariateUniformProposalDistribution()
: base_type(), lower_(0.0), upper_(1.0)
{}

double UnivariateUniformProposalDistribution::evaluate(const vector_type &x) const
{
	return k_ / (upper_ - lower_);
}

void UnivariateUniformProposalDistribution::sample(vector_type &sample) const
{
	// 0 <= x < 2 * pi
	sample[0] = swl::MathUtil::wrap(((double)std::rand() / RAND_MAX) * (upper_ - lower_) + lower_, 0.0, MathConstant::_2_PI);
}

void UnivariateUniformProposalDistribution::setParameters(const double lower, const double upper, const double k /*= 1.0*/)
{
	lower_ = lower;
	upper_ = upper;
	k_ = k;
}

void UnivariateUniformProposalDistribution::setSeed(const unsigned int seed)
{
	std::srand(seed);
}

//--------------------------------------------------------------------------
// find MAP estimate of multinomial using entropic prior.
bool computeMAPEstimateOfMultinomialUsingEntropicPrior(const std::vector<double> &omega, const double &z, std::vector<double> &theta, double &logLikelihood, const double terminationTolerance, const std::size_t maxIteration, const bool doesInitializeLambdaFirst /*= true*/)
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
