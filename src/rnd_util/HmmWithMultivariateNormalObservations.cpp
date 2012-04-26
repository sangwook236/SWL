#include "swl/Config.h"
#include "swl/rnd_util/HmmWithMultivariateNormalObservations.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <boost/numeric/ublas/blas.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/math/constants/constants.hpp>
#include <stdexcept>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

// [ref] swl/src/rnd_util/RndUtilLocalApi.cpp
double det_and_inv_by_lu(const boost::numeric::ublas::matrix<double> &m, boost::numeric::ublas::matrix<double> &inv);

HmmWithMultivariateNormalObservations::HmmWithMultivariateNormalObservations(const size_t K, const size_t D)
: base_type(K, D), mus_(K), sigmas_(K)  // 0-based index
{
	for (size_t k = 0; k < K; ++k)
	{
		mus_[k].resize(D);
		sigmas_[k].resize(D, D);
	}
}

HmmWithMultivariateNormalObservations::HmmWithMultivariateNormalObservations(const size_t K, const size_t D, const dvector_type &pi, const dmatrix_type &A, const std::vector<dvector_type> &mus, const std::vector<dmatrix_type> &sigmas)
: base_type(K, D, pi, A), mus_(mus), sigmas_(sigmas)
{
}

HmmWithMultivariateNormalObservations::~HmmWithMultivariateNormalObservations()
{
}

void HmmWithMultivariateNormalObservations::doEstimateObservationDensityParametersInMStep(const size_t N, const unsigned int state, const dmatrix_type &observations, dmatrix_type &gamma, const double denominatorA)
{
	// reestimate observation(emission) distribution in each state

	size_t n;
	const double denominator = denominatorA + gamma(N-1, state);
	const double factor = 0.999 / denominator;

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

void HmmWithMultivariateNormalObservations::doEstimateObservationDensityParametersInMStep(const std::vector<size_t> &Ns, const unsigned int state, const std::vector<dmatrix_type> &observationSequences, const std::vector<dmatrix_type> &gammas, const size_t R, const double denominatorA)
{
	// reestimate observation(emission) distribution in each state

	size_t n, r;
	double denominator = denominatorA;
	for (r = 0; r < R; ++r)
		denominator += gammas[r](Ns[r]-1, state);
	const double factor = 0.999 / denominator;

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

double HmmWithMultivariateNormalObservations::doEvaluateEmissionProbability(const unsigned int state, const boost::numeric::ublas::matrix_row<const dmatrix_type> &observation) const
{
	const dmatrix_type &sigma = sigmas_[state];
	dmatrix_type inv(sigma.size1(), sigma.size2());
	const double det = det_and_inv_by_lu(sigma, inv);

	const dvector_type x_mu(observation - mus_[state]);
	return std::exp(-0.5 * boost::numeric::ublas::inner_prod(x_mu, boost::numeric::ublas::prod(inv, x_mu))) / std::sqrt(std::pow(2.0 * boost::math::constants::pi<double>(), (double)D_) * det);
}

void HmmWithMultivariateNormalObservations::doGenerateObservationsSymbol(const unsigned int state, boost::numeric::ublas::matrix_row<dmatrix_type> &observation, const unsigned int seed /*= (unsigned int)-1*/) const
{
	// bivariate normal distribution
	if (2 == D_)
	{
		if ((unsigned int)-1 != seed)
		{
			// random number generator algorithms
			gsl_rng_default = gsl_rng_mt19937;
			//gsl_rng_default = gsl_rng_taus;
			gsl_rng_default_seed = (unsigned long)std::time(NULL);
		}

		const gsl_rng_type *T = gsl_rng_default;
		gsl_rng *r = gsl_rng_alloc(T);

		//
		const dvector_type &mu = mus_[state];
		const dmatrix_type &sigma = sigmas_[state];

		const double sigma_x = sigma(0, 0);
		const double sigma_y = sigma(1, 1);
		const double rho = sigma(0, 1) / std::sqrt(sigma_x * sigma_y);  // correlation coefficient

		double x = 0.0, y = 0.0;
		gsl_ran_bivariate_gaussian(r, sigma_x, sigma_y, rho, &x, &y);

		gsl_rng_free(r);

		observation[0] = mu[0] + x;
		observation[1] = mu[1] + y;
	}
	else
	{
		throw std::runtime_error("not yet implemented");
	}
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

	// K x D
	for (size_t k = 0; k < K_; ++k)
	{
		dvector_type &mu = mus_[k];

		for (d = 0; d < D_; ++d)
			stream >> mu[d];
	}

	stream >> dummy;
#if defined(__GNUC__)
	if (strcasecmp(dummy.c_str(), "sigma:") != 0)
#elif defined(_MSC_VER)
	if (_stricmp(dummy.c_str(), "sigma:") != 0)
#endif
		return false;

	// K x (D * D)
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

	// K x D
	stream << "mu:" << std::endl;
	for (k = 0; k < K_; ++k)
	{
		const dvector_type &mu = mus_[k];

		for (d = 0; d < D_; ++d)
			stream << mu[d] << ' ';
		stream << std::endl;
	}

	// K x (D * D)
	stream << "sigma:" << std::endl;
	for (k = 0; k < K_; ++k)
	{
		const dmatrix_type &sigma = sigmas_[k];

		for (d = 0; d < D_; ++d)
		{
			for (i = 0; i < D_; ++i)
				stream << sigma(d, i) << ' ';
			std::cout << "  ";
		}
		stream << std::endl;
	}

	return true;
}

void HmmWithMultivariateNormalObservations::doInitializeObservationDensity(const std::vector<double> &lowerBoundsOfObservationDensity, const std::vector<double> &upperBoundsOfObservationDensity)
{
	// PRECONDITIONS [] >>
	//	-. std::srand() had to be called before this function is called.

	// initialize the parameters of observation density
	const std::size_t numLowerBound = lowerBoundsOfObservationDensity.size();
	const std::size_t numUpperBound = upperBoundsOfObservationDensity.size();

	const std::size_t numParameters = K_ * (D_ + D_ * D_);  // the total number of parameters of observation density

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
		// TODO [check] >> all covariance matrices have to be symmetric positive definite.
		sigma = 0.5 * (sigma + boost::numeric::ublas::trans(sigma));
	}

	// POSTCONDITIONS [] >>
	//	-. all covariance matrices have to be symmetric positive definite.
}

}  // namespace swl
