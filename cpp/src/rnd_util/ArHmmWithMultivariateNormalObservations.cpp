#include "swl/Config.h"
#include "swl/rnd_util/ArHmmWithMultivariateNormalObservations.h"
#include "swl/math/MathConstant.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <boost/numeric/ublas/blas.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/math/distributions/normal.hpp>  // for normal distribution.
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/math/constants/constants.hpp>
#include <numeric>
#include <stdexcept>
#include <cassert>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

// [ref] swl/src/rnd_util/RndUtilLocalApi.cpp.
double det_and_inv_by_lu(const boost::numeric::ublas::matrix<double> &m, boost::numeric::ublas::matrix<double> &inv);
bool solve_linear_equations_by_lu(const boost::numeric::ublas::matrix<double> &m, boost::numeric::ublas::vector<double> &x);

ArHmmWithMultivariateNormalObservations::ArHmmWithMultivariateNormalObservations(const size_t K, const size_t D, const size_t P)
: base_type(K, D), P_(P), coeffs_(K, dmatrix_type(D, P, 0.0)), sigmas_(K, dvector_type(D, 0.0)),  // 0-based index.
  coeffs_conj_(),
  r_(NULL)
{
	assert(P_ > 0);
}

ArHmmWithMultivariateNormalObservations::ArHmmWithMultivariateNormalObservations(const size_t K, const size_t D, const size_t P, const dvector_type &pi, const dmatrix_type &A, const std::vector<dmatrix_type> &coeffs, const std::vector<dvector_type> &sigmas)
: base_type(K, D, pi, A), P_(P), coeffs_(coeffs), sigmas_(sigmas),
  coeffs_conj_(),
  r_(NULL)
{
	assert(P_ > 0);
}

ArHmmWithMultivariateNormalObservations::ArHmmWithMultivariateNormalObservations(const size_t K, const size_t D, const size_t P, const dvector_type *pi_conj, const dmatrix_type *A_conj, const std::vector<dmatrix_type> *coeffs_conj)
: base_type(K, D, pi_conj, A_conj), P_(P), coeffs_(K, dmatrix_type(D, P, 0.0)), sigmas_(K, dvector_type(D, 0.0)),
  coeffs_conj_(coeffs_conj),
  r_(NULL)
{
	// FIXME [modify] >>
	throw std::runtime_error("not yet implemented");

	assert(P_ > 0);
}

ArHmmWithMultivariateNormalObservations::~ArHmmWithMultivariateNormalObservations()
{
}

void ArHmmWithMultivariateNormalObservations::doEstimateObservationDensityParametersByML(const size_t N, const unsigned int state, const dmatrix_type &observations, const dmatrix_type &gamma, const double denominatorA)
{
	assert(N > P_);

	// M-step.
	// reestimate observation(emission) distribution in each state.

	const double eps = 1e-50;
	size_t n, p, j;
	int w;

	const double sumGamma = denominatorA + gamma(N-1, state);
	assert(std::fabs(sumGamma) >= eps);

	// TODO [check] >> is it good?
	const int W = int(P_);  // window size.
	const size_t L = 2 * W + 1;  // L consecutive samples of the observation signal.
	dvector_type winObs(L, 0.0);
	assert(L > P_);

	dmatrix_type autocovariance(N, P_ + 1, 0.0);  // autocovariance function.
	dvector_type autocovariance_bar(P_ + 1, 0.0);
	boost::numeric::ublas::matrix<double> A(P_, P_, 0.0);
	boost::numeric::ublas::vector<double> x(P_, 0.0);
	double meanWinObs;
	for (size_t d = 0; d < D_; ++d)
	{
		const dvector_type &observation_d = boost::numeric::ublas::matrix_column<const dmatrix_type>(observations, d);

		// calculate autocovariance functions.
		autocovariance.clear();
		for (n = 0; n < N; ++n)
		{
			for (w = -W; w <= W; ++w)
			{
				// TODO [check] >> which one is better?
				//winObs(w + W) = (int(n) + w < 0 || int(n) + w >= int(N)) ? 0.0 : observation_d(n + w);
				winObs(w + W) = int(n) + w < 0 ? observation_d(0) : (int(n) + w >= int(N) ? observation_d(N - 1) : observation_d(n + w));
			}

			meanWinObs = std::accumulate(winObs.begin(), winObs.end(), 0.0) / double(L);
			// zero mean observations.
			for (j = 0; j < L; ++j)
				winObs(j) -= meanWinObs;

			for (p = 0; p <= P_; ++p)
				for (j = 0; j < L - p; ++j)
					autocovariance(n, p) += winObs(j) * winObs(j + p);
		}

		autocovariance_bar.clear();
		for (p = 0; p <= P_; ++p)
		{
			for (n = 0; n < N; ++n)
				autocovariance_bar(p) += gamma(n, state) * autocovariance(n, p);
			autocovariance_bar(p) /= sumGamma;
		}

		// reestimate the autoregression coefficients.
		for (p = 0; p < P_; ++p)
		{
			x(p) = autocovariance_bar(p + 1);
			for (j = 0; j < p; ++j)
				A(p, j) = autocovariance_bar(p - j);
			for (j = p; j < P_; ++j)
				A(p, j) = autocovariance_bar(j - p);
		}

		if (solve_linear_equations_by_lu(A, x))
		{
#if defined(__GNUC__)
            // FIXME [check] >> is it correct?
			boost::numeric::ublas::matrix_row<dmatrix_type> coeff(boost::numeric::ublas::matrix_row<dmatrix_type>(coeffs_[state], d));
#else
			boost::numeric::ublas::matrix_row<dmatrix_type> &coeff = boost::numeric::ublas::matrix_row<dmatrix_type>(coeffs_[state], d);
#endif
			coeff = x;

			// reestimate the variances of the input noise process.
			double &sigma2 = sigmas_[state](d);
			sigma2 = autocovariance_bar(0);
			for (p = 0; p < P_; ++p)
				sigma2 -= coeff(p) * autocovariance_bar(p + 1);
			assert(sigma2 > 0.0);
		}
		else
		{
			assert(false);
		}
	}

	// POSTCONDITIONS [] >>
	//	-. all variances have to be positive.
}

void ArHmmWithMultivariateNormalObservations::doEstimateObservationDensityParametersByML(const std::vector<size_t> &Ns, const unsigned int state, const std::vector<dmatrix_type> &observationSequences, const std::vector<dmatrix_type> &gammas, const size_t R, const double denominatorA)
{
	// M-step.
	// reestimate observation(emission) distribution in each state.

	const double eps = 1e-50;
	size_t n, r, p, j;
	int w;

	double sumGamma = denominatorA;
	for (r = 0; r < R; ++r)
		sumGamma += gammas[r](Ns[r]-1, state);
	assert(std::fabs(sumGamma) >= eps);

	// TODO [check] >> is it good?
	const int W = int(P_);  // window size.
	const size_t L = 2 * W + 1;  // L consecutive samples of the observation signal.
	dvector_type winObs(L, 0.0);
	assert(L > P_);

	std::vector<dmatrix_type> autocovariances(R);  // autocovariance functions.
	dvector_type autocovariance_bar(P_ + 1, 0.0);
	boost::numeric::ublas::matrix<double> A(P_, P_, 0.0);
	boost::numeric::ublas::vector<double> x(P_, 0.0);
	double meanWinObs;
	for (size_t d = 0; d < D_; ++d)
	{
		for (r = 0; r < R; ++r)
		{
			assert(Ns[r] > P_);

			const dmatrix_type &observationr = observationSequences[r];
			//const dmatrix_type &gammar = gammas[r];
			dmatrix_type &autocovariancer = autocovariances[r];

			const dvector_type &observationr_d = boost::numeric::ublas::matrix_column<const dmatrix_type>(observationr, d);

			// calculate autocovariance functions.
			autocovariancer.resize(Ns[r], P_ + 1, false);
			autocovariancer.clear();
			for (n = 0; n < Ns[r]; ++n)
			{
				for (w = -W; w <= W; ++w)
				{
					// TODO [check] >> which one is better?
					//winObs(w + W) = (int(n) + w < 0 || int(n) + w >= int(Ns[r])) ? 0.0 : observationr_d(n + w);
					winObs(w + W) = int(n) + w < 0 ? observationr_d(0) : (int(n) + w >= int(Ns[r]) ? observationr_d(Ns[r] - 1) : observationr_d(n + w));
				}

				meanWinObs = std::accumulate(winObs.begin(), winObs.end(), 0.0) / double(L);
				// zero mean observations.
				for (j = 0; j < L; ++j)
					winObs(j) -= meanWinObs;

				for (p = 0; p <= P_; ++p)
					for (j = 0; j < L - p; ++j)
						autocovariancer(n, p) += winObs(j) * winObs(j + p);
			}
		}

		autocovariance_bar.clear();
		for (p = 0; p <= P_; ++p)
		{
			for (r = 0; r < R; ++r)
			{
				const dmatrix_type &gammar = gammas[r];
				const dmatrix_type &autocovariancer = autocovariances[r];

				for (n = 0; n < Ns[r]; ++n)
					autocovariance_bar(p) += gammar(n, state) * autocovariancer(n, p);
			}

			autocovariance_bar(p) /= sumGamma;
		}

		// reestimate the autoregression coefficients.
		for (p = 0; p < P_; ++p)
		{
			x(p) = autocovariance_bar(p + 1);
			for (j = 0; j < p; ++j)
				A(p, j) = autocovariance_bar(p - j);
			for (j = p; j < P_; ++j)
				A(p, j) = autocovariance_bar(j - p);
		}

		if (solve_linear_equations_by_lu(A, x))
		{
#if defined(__GNUC__)
            // FIXME [check] >> is it correct?
			boost::numeric::ublas::matrix_row<dmatrix_type> coeff(boost::numeric::ublas::matrix_row<dmatrix_type>(coeffs_[state], d));
#else
			boost::numeric::ublas::matrix_row<dmatrix_type> &coeff = boost::numeric::ublas::matrix_row<dmatrix_type>(coeffs_[state], d);
#endif
			coeff = x;

			// reestimate the variances of the input noise process.
			double &sigma2 = sigmas_[state](d);
			sigma2 = autocovariance_bar(0);
			for (p = 0; p < P_; ++p)
				sigma2 -= coeff(p) * autocovariance_bar(p + 1);
			assert(sigma2 > 0.0);
		}
		else
		{
			assert(false);
		}
	}

	// POSTCONDITIONS [] >>
	//	-. all variances have to be positive.
}

void ArHmmWithMultivariateNormalObservations::doEstimateObservationDensityParametersByMAPUsingConjugatePrior(const size_t N, const unsigned int state, const dmatrix_type &observations, const dmatrix_type &gamma, const double denominatorA)
{
	// FIXME [modify] >>
	throw std::runtime_error("not yet implemented");

#if 0
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
#endif
}

void ArHmmWithMultivariateNormalObservations::doEstimateObservationDensityParametersByMAPUsingConjugatePrior(const std::vector<size_t> &Ns, const unsigned int state, const std::vector<dmatrix_type> &observationSequences, const std::vector<dmatrix_type> &gammas, const size_t R, const double denominatorA)
{
	// FIXME [modify] >>
	throw std::runtime_error("not yet implemented");

#if 0
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
#endif
}

void ArHmmWithMultivariateNormalObservations::doEstimateObservationDensityParametersByMAPUsingEntropicPrior(const size_t N, const unsigned int state, const dmatrix_type &observations, const dmatrix_type &gamma, const double /*z*/, const bool /*doesTrimParameter*/, const double /*terminationTolerance*/, const size_t /*maxIteration*/, const double denominatorA)
{
	doEstimateObservationDensityParametersByML(N, state, observations, gamma, denominatorA);
}

void ArHmmWithMultivariateNormalObservations::doEstimateObservationDensityParametersByMAPUsingEntropicPrior(const std::vector<size_t> &Ns, const unsigned int state, const std::vector<dmatrix_type> &observationSequences, const std::vector<dmatrix_type> &gammas, const double /*z*/, const bool /*doesTrimParameter*/, const double /*terminationTolerance*/, const size_t /*maxIteration*/, const size_t R, const double denominatorA)
{
	doEstimateObservationDensityParametersByML(Ns, state, observationSequences, gammas, R, denominatorA);
}

double ArHmmWithMultivariateNormalObservations::doEvaluateEmissionProbability(const unsigned int state, const dvector_type &observation) const
{
	assert(false);

	dvector_type M(D_, 0.0);
	dmatrix_type S(D_, D_, 0.0);

	//const dmatrix_type &coeff = coeffs_[state];
	const dvector_type &sigma2 = sigmas_[state];
	for (size_t d = 0; d < D_; ++d)
	{
		for (size_t p = 0; p < P_; ++p)
		{
			// FIXME [check] > is it correct?
			// TODO [check] >> which one is better?
			M(d) = 0.0;  //coeff(d, p) * 0.0;
			//M(d) = coeff(d, p) * observation(d);
		}
		S(d, d) = sigma2(d);
	}

	dmatrix_type inv(S.size1(), S.size2());
	const double det = det_and_inv_by_lu(S, inv);
	assert(det > 0.0);

	const dvector_type x_mu(observation - M);
	return std::exp(-0.5 * boost::numeric::ublas::inner_prod(x_mu, boost::numeric::ublas::prod(inv, x_mu))) / std::sqrt(std::pow(MathConstant::_2_PI, (double)D_) * det);
}

double ArHmmWithMultivariateNormalObservations::doEvaluateEmissionProbability(const unsigned int state, const size_t n, const dmatrix_type &observations) const
{
	dvector_type M(D_, 0.0);
	dmatrix_type S(D_, D_, 0.0);

	const dmatrix_type &coeff = coeffs_[state];
	const dvector_type &sigma2 = sigmas_[state];
	for (size_t d = 0; d < D_; ++d)
	{
		for (size_t p = 0; p < P_; ++p)
		{
			// TODO [check] >> which one is better?
			//M(d) = coeff(d, p) * ((int(n) - int(p) - 1 < 0) ? 0.0 : observations(n-p-1, 0));
			M(d) = coeff(d, p) * ((int(n) - int(p) - 1 < 0) ? observations(0, 0) : observations(n-p-1, 0));
		}
		S(d, d) = sigma2(d);
	}

	dmatrix_type inv(S.size1(), S.size2());
	const double det = det_and_inv_by_lu(S, inv);
	assert(det > 0.0);

	const dvector_type x_mu(boost::numeric::ublas::matrix_row<const dmatrix_type>(observations, n) - M);
	return std::exp(-0.5 * boost::numeric::ublas::inner_prod(x_mu, boost::numeric::ublas::prod(inv, x_mu))) / std::sqrt(std::pow(MathConstant::_2_PI, (double)D_) * det);
}

void ArHmmWithMultivariateNormalObservations::doGenerateObservationsSymbol(const unsigned int state, const size_t n, dmatrix_type &observations) const
{
	assert(NULL != r_);

	// bivariate normal distribution.
	if (2 == D_)
	{
		dvector_type M(D_, 0.0);
		dmatrix_type S(D_, D_, 0.0);

		const dmatrix_type &coeff = coeffs_[state];
		const dvector_type &sigma2 = sigmas_[state];
		for (size_t d = 0; d < D_; ++d)
		{
			for (size_t p = 0; p < P_; ++p)
			{
				// TODO [check] >> which one is better?
				//M(d) = coeff(d, p) * ((int(n) - int(p) - 1 < 0) ? 0.0 : observations(n-p-1, d));
				M(d) = coeff(d, p) * ((int(n) - int(p) - 1 < 0) ? observations(0, d) : observations(n-p-1, d));
			}
			S(d, d) = sigma2(d);
		}

		const double sigma_x = std::sqrt(S(0, 0));  // sigma_x = sqrt(cov_xx).
		const double sigma_y = std::sqrt(S(1, 1));  // sigma_y = sqrt(cov_yy).
		const double rho = S(0, 1) / (sigma_x * sigma_y);  // correlation coefficient: rho = cov_xy / (sigma_x * sigma_y).

		double x = 0.0, y = 0.0;
		gsl_ran_bivariate_gaussian(r_, sigma_x, sigma_y, rho, &x, &y);

		observations(n, 0) = M[0] + x;
		observations(n, 1) = M[1] + y;
	}
	else
	{
		throw std::runtime_error("not yet implemented");
	}
}

void ArHmmWithMultivariateNormalObservations::doInitializeRandomSampleGeneration(const unsigned int seed /*= (unsigned int)-1*/) const
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

void ArHmmWithMultivariateNormalObservations::doFinalizeRandomSampleGeneration() const
{
	gsl_rng_free(r_);
	r_ = NULL;
}

bool ArHmmWithMultivariateNormalObservations::doReadObservationDensity(std::istream &stream)
{
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

	// TODO [check] >>
	size_t P;
	stream >> dummy >> P;  // the order of autoregressive model.
#if defined(__GNUC__)
	if (strcasecmp(dummy.c_str(), "P=") != 0 || P_ != P)
#elif defined(_MSC_VER)
	if (_stricmp(dummy.c_str(), "P=") != 0 || P_ != P)
#endif
		return false;

	stream >> dummy;
#if defined(__GNUC__)
	if (strcasecmp(dummy.c_str(), "coeff:") != 0)
#elif defined(_MSC_VER)
	if (_stricmp(dummy.c_str(), "coeff:") != 0)
#endif
		return false;

	size_t k, d, p;

	// K x D x P.
	for (k = 0; k < K_; ++k)
	{
		dmatrix_type &coeff = coeffs_[k];

		for (d = 0; d < D_; ++d)
			for (p = 0; p < P_; ++p)
				stream >> coeff(d, p);
	}

	stream >> dummy;
#if defined(__GNUC__)
	if (strcasecmp(dummy.c_str(), "sigma:") != 0)
#elif defined(_MSC_VER)
	if (_stricmp(dummy.c_str(), "sigma:") != 0)
#endif
		return false;

	// K x D.
	for (k = 0; k < K_; ++k)
	{
		dvector_type &sigma2 = sigmas_[k];

		for (d = 0; d < D_; ++d)
			stream >> sigma2(d);
	}

	return true;
}

bool ArHmmWithMultivariateNormalObservations::doWriteObservationDensity(std::ostream &stream) const
{
	stream << "ar multivariate normal:" << std::endl;

	stream << "P= " << P_ << std::endl;  // the order of autoregressive model.

	size_t k, d, p;

	// K x D x P.
	stream << "coeff:" << std::endl;
	for (k = 0; k < K_; ++k)
	{
		const dmatrix_type &coeff = coeffs_[k];

		for (d = 0; d < D_; ++d)
		{
			for (p = 0; p < P_; ++p)
				stream << coeff(d, p) << ' ';
			stream << "  ";
		}
		stream << std::endl;
	}

	// K x D.
	stream << "sigma:" << std::endl;
	for (k = 0; k < K_; ++k)
	{
		const dvector_type &sigma2 = sigmas_[k];

		for (d = 0; d < D_; ++d)
			stream << sigma2(d) << ' ';
		stream << std::endl;
	}

	return true;
}

void ArHmmWithMultivariateNormalObservations::doInitializeObservationDensity(const std::vector<double> &lowerBoundsOfObservationDensity, const std::vector<double> &upperBoundsOfObservationDensity)
{
	// PRECONDITIONS [] >>
	//	-. std::srand() has to be called before this function is called.

	// initialize the parameters of observation density.
	const std::size_t numLowerBound = lowerBoundsOfObservationDensity.size();
	const std::size_t numUpperBound = upperBoundsOfObservationDensity.size();

	const std::size_t numParameters = K_ * D_ * P_ + K_ * D_;  // the total number of parameters of observation density.

	assert(numLowerBound == numUpperBound);
	assert(1 == numLowerBound || numParameters == numLowerBound);

	if (1 == numLowerBound)
	{
		const double lb = lowerBoundsOfObservationDensity[0], ub = upperBoundsOfObservationDensity[0];
		size_t k, d, p;
		for (k = 0; k < K_; ++k)
		{
			dmatrix_type &coeff = coeffs_[k];
			dvector_type &sigma2 = sigmas_[k];
			for (d = 0; d < D_; ++d)
			{
				for (p = 0; p < P_; ++p)
					coeff(d, p) = ((double)std::rand() / RAND_MAX) * (ub - lb) + lb;
				sigma2(d) = ((double)std::rand() / RAND_MAX) * (ub - lb) + lb;
			}
		}
	}
	else if (numParameters == numLowerBound)
	{
		size_t k, d, p, idx = 0;
		for (k = 0; k < K_; ++k)
		{
			dmatrix_type &coeff = coeffs_[k];
			for (d = 0; d < D_; ++d)
				for (p = 0; p < P_; ++p, ++idx)
					coeff(d, p) = ((double)std::rand() / RAND_MAX) * (upperBoundsOfObservationDensity[idx] - lowerBoundsOfObservationDensity[idx]) + lowerBoundsOfObservationDensity[idx];
		}
		for (k = 0; k < K_; ++k)
		{
			dvector_type &sigma2 = sigmas_[k];
			for (d = 0; d < D_; ++d, ++idx)
				sigma2(d) = ((double)std::rand() / RAND_MAX) * (upperBoundsOfObservationDensity[idx] - lowerBoundsOfObservationDensity[idx]) + lowerBoundsOfObservationDensity[idx];
		}
	}

	for (size_t k = 0; k < K_; ++k)
	{
		dvector_type &sigma2 = sigmas_[k];

		// all variances have to be symmetric positive definite.
		for (size_t d = 0; d < D_; ++d)
			if (sigma2(d) < 0.0)
				sigma2(d) = -sigma2(d);
	}

	// POSTCONDITIONS [] >>
	//	-. all variances have to be symmetric positive definite.
}

}  // namespace swl
