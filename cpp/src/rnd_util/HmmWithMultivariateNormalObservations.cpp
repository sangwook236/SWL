#include "swl/Config.h"
#include "swl/rnd_util/HmmWithMultivariateNormalObservations.h"
#include "swl/math/MathConstant.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <boost/numeric/ublas/blas.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/math/constants/constants.hpp>
#include <stdexcept>
#include <cassert>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

// [ref] swl/src/rnd_util/RndUtilLocalApi.cpp.
double det_and_inv_by_lu(const boost::numeric::ublas::matrix<double> &m, boost::numeric::ublas::matrix<double> &inv);

HmmWithMultivariateNormalObservations::HmmWithMultivariateNormalObservations(const size_t K, const size_t D)
: base_type(K, D), mus_(K), sigmas_(K),  // 0-based index.
  mus_conj_(), betas_conj_(), sigmas_conj_(), nus_conj_(),
  r_(NULL)
{
	for (size_t k = 0; k < K; ++k)
	{
		mus_[k].resize(D);
		sigmas_[k].resize(D, D);
	}
}

HmmWithMultivariateNormalObservations::HmmWithMultivariateNormalObservations(const size_t K, const size_t D, const dvector_type &pi, const dmatrix_type &A, const std::vector<dvector_type> &mus, const std::vector<dmatrix_type> &sigmas)
: base_type(K, D, pi, A), mus_(mus), sigmas_(sigmas),
  mus_conj_(), betas_conj_(), sigmas_conj_(), nus_conj_(),
  r_(NULL)
{
}

HmmWithMultivariateNormalObservations::HmmWithMultivariateNormalObservations(const size_t K, const size_t D, const dvector_type *pi_conj, const dmatrix_type *A_conj, const std::vector<dvector_type> *mus_conj, const dvector_type *betas_conj, const std::vector<dmatrix_type> *sigmas_conj, const dvector_type *nus_conj)
: base_type(K, D, pi_conj, A_conj), mus_(K), sigmas_(K),
  mus_conj_(mus_conj), betas_conj_(betas_conj), sigmas_conj_(sigmas_conj), nus_conj_(nus_conj),
  r_(NULL)
{
}

HmmWithMultivariateNormalObservations::~HmmWithMultivariateNormalObservations()
{
}

void HmmWithMultivariateNormalObservations::doEstimateObservationDensityParametersByML(const size_t N, const unsigned int state, const dmatrix_type &observations, const dmatrix_type &gamma, const double denominatorA)
{
	// M-step.
	// reestimate observation(emission) distribution in each state.

	const double eps = 1e-50;
	size_t n;
	const double sumGamma = denominatorA + gamma(N-1, state);
	assert(std::fabs(sumGamma) >= eps);
	const double factor = 0.999 / sumGamma;

	//
	dvector_type &mu = mus_[state];
	mu.clear();
	for (n = 0; n < N; ++n)
		mu += gamma(n, state) * boost::numeric::ublas::matrix_row<const dmatrix_type>(observations, n);
	//mu = mu * factor + boost::numeric::ublas::scalar_vector<double>(mu.size(), 0.001);
	mu = mu * factor + boost::numeric::ublas::scalar_vector<double>(D_, 0.001);

	//
	dmatrix_type &sigma = sigmas_[state];
	sigma.clear();
	for (n = 0; n < N; ++n)
		boost::numeric::ublas::blas_2::sr(sigma, gamma(n, state), boost::numeric::ublas::matrix_row<const dmatrix_type>(observations, n) - mu);
	sigma = 0.5 * (sigma + boost::numeric::ublas::trans(sigma));

	//sigma = sigma * factor + boost::numeric::ublas::scalar_matrix<double>(sigma.size1(), sigma.size2(), 0.001);
	sigma = sigma * factor + boost::numeric::ublas::scalar_matrix<double>(D_, D_, 0.001);

	// POSTCONDITIONS [] >>
	//	-. all covariance matrices have to be symmetric positive definite.
}

void HmmWithMultivariateNormalObservations::doEstimateObservationDensityParametersByML(const std::vector<size_t> &Ns, const unsigned int state, const std::vector<dmatrix_type> &observationSequences, const std::vector<dmatrix_type> &gammas, const size_t R, const double denominatorA)
{
	// M-step.
	// reestimate observation(emission) distribution in each state.

	const double eps = 1e-50;
	size_t n, r;
	double sumGamma = denominatorA;
	for (r = 0; r < R; ++r)
		sumGamma += gammas[r](Ns[r]-1, state);
	assert(std::fabs(sumGamma) >= eps);
	const double factor = 0.999 / sumGamma;

	//
	dvector_type &mu = mus_[state];
	mu.clear();
	for (r = 0; r < R; ++r)
	{
		const dmatrix_type &gammar = gammas[r];
		const dmatrix_type &observationr = observationSequences[r];

		for (n = 0; n < Ns[r]; ++n)
			mu += gammar(n, state) * boost::numeric::ublas::matrix_row<const dmatrix_type>(observationr, n);
	}
	//mu = factor * mu + boost::numeric::ublas::scalar_vector<double>(mu.size(), 0.001);
	mu = factor * mu + boost::numeric::ublas::scalar_vector<double>(D_, 0.001);

	//
	dmatrix_type &sigma = sigmas_[state];
	sigma.clear();
	for (r = 0; r < R; ++r)
	{
		const dmatrix_type &gammar = gammas[r];
		const dmatrix_type &observationr = observationSequences[r];

		for (n = 0; n < Ns[r]; ++n)
			boost::numeric::ublas::blas_2::sr(sigma, gammar(n, state), boost::numeric::ublas::matrix_row<const dmatrix_type>(observationr, n) - mu);
	}
	sigma = 0.5 * (sigma + boost::numeric::ublas::trans(sigma));
	
	//sigma = sigma * factor + boost::numeric::ublas::scalar_matrix<double>(sigma.size1(), sigma.size2(), 0.001);
	sigma = sigma * factor + boost::numeric::ublas::scalar_matrix<double>(D_, D_, 0.001);

	// POSTCONDITIONS [] >>
	//	-. all covariance matrices have to be symmetric positive definite.
}

void HmmWithMultivariateNormalObservations::doEstimateObservationDensityParametersByMAPUsingConjugatePrior(const size_t N, const unsigned int state, const dmatrix_type &observations, const dmatrix_type &gamma, const double denominatorA)
{
	// M-step.
	// reestimate observation(emission) distribution in each state.

	size_t n;
	const double sumGamma = denominatorA + gamma(N-1, state);
	//assert(std::fabs(sumGamma) >= eps);
	const double factorMu = 0.999 / (sumGamma + (*betas_conj_)[state]);
	const double factorSigma = 0.999 / (sumGamma + (*nus_conj_)[state] - D_);

	//
	dvector_type &mu = mus_[state];
	mu = (*betas_conj_)[state] * (*mus_conj_)[state];
	for (n = 0; n < N; ++n)
		mu += gamma(n, state) * boost::numeric::ublas::matrix_row<const dmatrix_type>(observations, n);
	//mu = mu * factorMu + boost::numeric::ublas::scalar_vector<double>(mu.size(), 0.001);
	mu = mu * factorMu + boost::numeric::ublas::scalar_vector<double>(D_, 0.001);

	//
	dmatrix_type &sigma = sigmas_[state];
	sigma = (*sigmas_conj_)[state];
	boost::numeric::ublas::blas_2::sr(sigma, (*betas_conj_)[state], mu - (*mus_conj_)[state]);
	for (n = 0; n < N; ++n)
		boost::numeric::ublas::blas_2::sr(sigma, gamma(n, state), boost::numeric::ublas::matrix_row<const dmatrix_type>(observations, n) - mu);
	sigma = 0.5 * (sigma + boost::numeric::ublas::trans(sigma));

	//sigma = sigma * factorSigma + boost::numeric::ublas::scalar_matrix<double>(sigma.size1(), sigma.size2(), 0.001);
	sigma = sigma * factorSigma + boost::numeric::ublas::scalar_matrix<double>(D_, D_, 0.001);

	// POSTCONDITIONS [] >>
	//	-. all covariance matrices have to be symmetric positive definite.
}

void HmmWithMultivariateNormalObservations::doEstimateObservationDensityParametersByMAPUsingConjugatePrior(const std::vector<size_t> &Ns, const unsigned int state, const std::vector<dmatrix_type> &observationSequences, const std::vector<dmatrix_type> &gammas, const size_t R, const double denominatorA)
{
	// M-step.
	// reestimate observation(emission) distribution in each state.

	size_t n, r;
	double sumGamma = denominatorA;
	for (r = 0; r < R; ++r)
		sumGamma += gammas[r](Ns[r]-1, state);
	//assert(std::fabs(sumGamma) >= eps);
	const double factorMu = 0.999 / (sumGamma + (*betas_conj_)(state));
	const double factorSigma = 0.999 / (sumGamma + (*nus_conj_)(state) - D_);

	//
	dvector_type &mu = mus_[state];
	mu = (*betas_conj_)(state) * (*mus_conj_)[state];
	for (r = 0; r < R; ++r)
	{
		const dmatrix_type &gammar = gammas[r];
		const dmatrix_type &observationr = observationSequences[r];

		for (n = 0; n < Ns[r]; ++n)
			mu += gammar(n, state) * boost::numeric::ublas::matrix_row<const dmatrix_type>(observationr, n);
	}
	//mu = mu * factorMu + boost::numeric::ublas::scalar_vector<double>(mu.size(), 0.001);
	mu = mu * factorMu + boost::numeric::ublas::scalar_vector<double>(D_, 0.001);

	//
	dmatrix_type &sigma = sigmas_[state];
	sigma = (*sigmas_conj_)[state];
	boost::numeric::ublas::blas_2::sr(sigma, (*betas_conj_)(state), mu - (*mus_conj_)[state]);
	for (r = 0; r < R; ++r)
	{
		const dmatrix_type &gammar = gammas[r];
		const dmatrix_type &observationr = observationSequences[r];

		for (n = 0; n < Ns[r]; ++n)
			boost::numeric::ublas::blas_2::sr(sigma, gammar(n, state), boost::numeric::ublas::matrix_row<const dmatrix_type>(observationr, n) - mu);
	}
	sigma = 0.5 * (sigma + boost::numeric::ublas::trans(sigma));
	
	//sigma = sigma * factorSigma + boost::numeric::ublas::scalar_matrix<double>(sigma.size1(), sigma.size2(), 0.001);
	sigma = sigma * factorSigma + boost::numeric::ublas::scalar_matrix<double>(D_, D_, 0.001);

	// POSTCONDITIONS [] >>
	//	-. all covariance matrices have to be symmetric positive definite.
}

void HmmWithMultivariateNormalObservations::doEstimateObservationDensityParametersByMAPUsingEntropicPrior(const size_t N, const unsigned int state, const dmatrix_type &observations, const dmatrix_type &gamma, const double /*z*/, const bool /*doesTrimParameter*/, const double /*terminationTolerance*/, const size_t /*maxIteration*/, const double denominatorA)
{
	doEstimateObservationDensityParametersByML(N, state, observations, gamma, denominatorA);
}

void HmmWithMultivariateNormalObservations::doEstimateObservationDensityParametersByMAPUsingEntropicPrior(const std::vector<size_t> &Ns, const unsigned int state, const std::vector<dmatrix_type> &observationSequences, const std::vector<dmatrix_type> &gammas, const double /*z*/, const bool /*doesTrimParameter*/, const double /*terminationTolerance*/, const size_t /*maxIteration*/, const size_t R, const double denominatorA)
{
	doEstimateObservationDensityParametersByML(Ns, state, observationSequences, gammas, R, denominatorA);
}

double HmmWithMultivariateNormalObservations::doEvaluateEmissionProbability(const unsigned int state, const dvector_type &observation) const
{
	const dmatrix_type &sigma = sigmas_[state];
	dmatrix_type inv(sigma.size1(), sigma.size2());
	const double det = det_and_inv_by_lu(sigma, inv);
	assert(det > 0.0);

	const dvector_type x_mu(observation - mus_[state]);
	return std::exp(-0.5 * boost::numeric::ublas::inner_prod(x_mu, boost::numeric::ublas::prod(inv, x_mu))) / std::sqrt(std::pow(MathConstant::_2_PI, (double)D_) * det);
}

void HmmWithMultivariateNormalObservations::doGenerateObservationsSymbol(const unsigned int state, const size_t n, dmatrix_type &observations) const
{
	assert(NULL != r_);

	// bivariate normal distribution.
	if (2 == D_)
	{
		const dvector_type &mu = mus_[state];
		const dmatrix_type &cov = sigmas_[state];

		const double sigma_x = std::sqrt(cov(0, 0));  // sigma_x = sqrt(cov_xx).
		const double sigma_y = std::sqrt(cov(1, 1));  // sigma_y = sqrt(cov_yy).
		const double rho = cov(0, 1) / (sigma_x * sigma_y);  // correlation coefficient: rho = cov_xy / (sigma_x * sigma_y).

		double x = 0.0, y = 0.0;
		gsl_ran_bivariate_gaussian(r_, sigma_x, sigma_y, rho, &x, &y);

		observations(n, 0) = mu[0] + x;
		observations(n, 1) = mu[1] + y;
	}
	else
	{
		throw std::runtime_error("not yet implemented");
	}
}

void HmmWithMultivariateNormalObservations::doInitializeRandomSampleGeneration(const unsigned int seed /*= (unsigned int)-1*/) const
{
	if ((unsigned int)-1 != seed)
	{
		// random number generator algorithms.
		gsl_rng_default = gsl_rng_mt19937;
		//gsl_rng_default = gsl_rng_taus;
		gsl_rng_default_seed = seed;
	}

	const gsl_rng_type *T = gsl_rng_default;
	r_ = gsl_rng_alloc(T);
}

void HmmWithMultivariateNormalObservations::doFinalizeRandomSampleGeneration() const
{
	gsl_rng_free(r_);
	r_ = NULL;
}

bool HmmWithMultivariateNormalObservations::doReadObservationDensity(std::istream &stream)
{
	std::string dummy;
	stream >> dummy;
#if defined(__GNUC__)
	if (strcasecmp(dummy.c_str(), "multivariate") != 0)
#elif defined(_MSC_VER)
	if (_stricmp(dummy.c_str(), "multivariate") != 0)
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

	size_t d, i;

	// K x D.
	for (size_t k = 0; k < K_; ++k)
	{
		dvector_type &mu = mus_[k];

		for (d = 0; d < D_; ++d)
			stream >> mu[d];
	}

	stream >> dummy;
#if defined(__GNUC__)
	if (strcasecmp(dummy.c_str(), "covariance:") != 0)
#elif defined(_MSC_VER)
	if (_stricmp(dummy.c_str(), "covariance:") != 0)
#endif
		return false;

	// K x (D * D).
	for (size_t k = 0; k < K_; ++k)
	{
		dmatrix_type &sigma = sigmas_[k];
	
		for (d = 0; d < D_; ++d)
			for (i = 0; i < D_; ++i)
				stream >> sigma(d, i);
	}

	return true;
}

bool HmmWithMultivariateNormalObservations::doWriteObservationDensity(std::ostream &stream) const
{
	stream << "multivariate normal:" << std::endl;

	size_t i, k, d;

	// K x D.
	stream << "mu:" << std::endl;
	for (k = 0; k < K_; ++k)
	{
		const dvector_type &mu = mus_[k];

		for (d = 0; d < D_; ++d)
			stream << mu[d] << ' ';
		stream << std::endl;
	}

	// K x (D * D).
	stream << "covariance:" << std::endl;
	for (k = 0; k < K_; ++k)
	{
		const dmatrix_type &sigma = sigmas_[k];

		for (d = 0; d < D_; ++d)
		{
			for (i = 0; i < D_; ++i)
				stream << sigma(d, i) << ' ';
			stream << "  ";
		}
		stream << std::endl;
	}

	return true;
}

void HmmWithMultivariateNormalObservations::doInitializeObservationDensity(const std::vector<double> &lowerBoundsOfObservationDensity, const std::vector<double> &upperBoundsOfObservationDensity)
{
	// PRECONDITIONS [] >>
	//	-. std::srand() has to be called before this function is called.

	// initialize the parameters of observation density.
	const std::size_t numLowerBound = lowerBoundsOfObservationDensity.size();
	const std::size_t numUpperBound = upperBoundsOfObservationDensity.size();

	const std::size_t numParameters = K_ * (D_ + D_ * D_);  // the total number of parameters of observation density.

	assert(numLowerBound == numUpperBound);
	assert(1 == numLowerBound || numParameters == numLowerBound);

	if (1 == numLowerBound)
	{
		const double lb = lowerBoundsOfObservationDensity[0], ub = upperBoundsOfObservationDensity[0];
		size_t d, i;
		for (size_t k = 0; k < K_; ++k)
		{
			dvector_type &mu = mus_[k];
			dmatrix_type &sigma = sigmas_[k];
			for (d = 0; d < D_; ++d)
			{
				mu[d] = ((double)std::rand() / RAND_MAX) * (ub - lb) + lb;
				for (i = 0; i < D_; ++i)
					sigma(d, i) = ((double)std::rand() / RAND_MAX) * (ub - lb) + lb;
			}
		}
	}
	else if (numParameters == numLowerBound)
	{
		size_t k, d, i, idx = 0;
		for (k = 0; k < K_; ++k)
		{
			dvector_type &mu = mus_[k];
			for (d = 0; d < D_; ++d, ++idx)
				mu[d] = ((double)std::rand() / RAND_MAX) * (upperBoundsOfObservationDensity[idx] - lowerBoundsOfObservationDensity[idx]) + lowerBoundsOfObservationDensity[idx];
		}
		for (k = 0; k < K_; ++k)
		{
			dmatrix_type &sigma = sigmas_[k];
			for (d = 0; d < D_; ++d)
				for (i = 0; i < D_; ++i, ++idx)
					sigma(d, i) = ((double)std::rand() / RAND_MAX) * (upperBoundsOfObservationDensity[idx] - lowerBoundsOfObservationDensity[idx]) + lowerBoundsOfObservationDensity[idx];
		}
	}

	for (size_t k = 0; k < K_; ++k)
	{
		dmatrix_type &sigma = sigmas_[k];

		// all covariance matrices have to be symmetric positive definite.
		boost::numeric::ublas::blas_3::srk(sigma, 0.0, 1.0, sigma);  // m1 = t1 * m1 + t2 * (m2 * m2^T).
		sigma = 0.5 * (sigma + boost::numeric::ublas::trans(sigma));
	}

	// POSTCONDITIONS [] >>
	//	-. all covariance matrices have to be symmetric positive definite.
}

}  // namespace swl
