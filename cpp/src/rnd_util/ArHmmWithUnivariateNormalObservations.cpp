#include "swl/Config.h"
#include "swl/rnd_util/ArHmmWithUnivariateNormalObservations.h"
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/math/distributions/normal.hpp>  // for normal distribution.
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <numeric>
#include <ctime>
#include <stdexcept>
#include <cassert>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

// [ref] swl/src/rnd_util/RndUtilLocalApi.cpp.
bool solve_linear_equations_by_lu(const boost::numeric::ublas::matrix<double> &m, boost::numeric::ublas::vector<double> &x);

ArHmmWithUnivariateNormalObservations::ArHmmWithUnivariateNormalObservations(const size_t K, const size_t P)
: base_type(K, 1), P_(P), coeffs_(K, dvector_type(P, 0.0)), sigmas_(K, 0.0),  // 0-based index.
  coeffs_conj_(),
  baseGenerator_()
{
	assert(P_ > 0);
}

ArHmmWithUnivariateNormalObservations::ArHmmWithUnivariateNormalObservations(const size_t K, const size_t P, const dvector_type &pi, const dmatrix_type &A, const std::vector<dvector_type> &coeffs, const std::vector<double> &sigmas)
: base_type(K, 1, pi, A), P_(P), coeffs_(coeffs), sigmas_(sigmas),
  coeffs_conj_(),
  baseGenerator_()
{
	assert(P_ > 0);
}

ArHmmWithUnivariateNormalObservations::ArHmmWithUnivariateNormalObservations(const size_t K, const size_t P, const dvector_type *pi_conj, const dmatrix_type *A_conj, const std::vector<dvector_type> *coeffs_conj)
: base_type(K, 1, pi_conj, A_conj), P_(P), coeffs_(K, dvector_type(P, 0.0)), sigmas_(K, 0.0),
  coeffs_conj_(coeffs_conj),
  baseGenerator_()
{
	// FIXME [modify] >>
	throw std::runtime_error("Not yet implemented");

	assert(P_ > 0);
}

ArHmmWithUnivariateNormalObservations::~ArHmmWithUnivariateNormalObservations()
{
}

void ArHmmWithUnivariateNormalObservations::doEstimateObservationDensityParametersByML(const size_t N, const unsigned int state, const dmatrix_type &observations, const dmatrix_type &gamma, const double denominatorA)
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

	// calculate autocovariance functions.
	dmatrix_type autocovariance(N, P_ + 1, 0.0);  // autocovariance function.
	double meanWinObs;
	for (n = 0; n < N; ++n)
	{
		for (w = -W; w <= W; ++w)
		{
			// TODO [check] >> which one is better?
			//winObs(w + W) = (int(n) + w < 0 || int(n) + w >= int(N)) ? 0.0 : observations(n + w, 0);
			winObs(w + W) = int(n) + w < 0 ? observations(0, 0) : (int(n) + w >= int(N) ? observations(N - 1, 0) : observations(n + w, 0));
		}

		meanWinObs = std::accumulate(winObs.begin(), winObs.end(), 0.0) / double(L);
		// zero mean observations.
		for (j = 0; j < L; ++j)
			winObs(j) -= meanWinObs;

		for (p = 0; p <= P_; ++p)
			for (j = 0; j < L - p; ++j)
				autocovariance(n, p) += winObs(j) * winObs(j + p);
	}

	dvector_type autocovariance_bar(P_ + 1, 0.0);
	for (p = 0; p <= P_; ++p)
	{
		for (n = 0; n < N; ++n)
			autocovariance_bar(p) += gamma(n, state) * autocovariance(n, p);
		autocovariance_bar(p) /= sumGamma;
	}

	// reestimate the autoregression coefficients.
	boost::numeric::ublas::matrix<double> A(P_, P_, 0.0);
	boost::numeric::ublas::vector<double> x(P_, 0.0);
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
		dvector_type &coeff = coeffs_[state];
		coeff = x;

		// reestimate the variances of the input noise process.
		double &sigma2 = sigmas_[state];
		sigma2 = autocovariance_bar(0);
		for (p = 0; p < P_; ++p)
			sigma2 -= coeff(p) * autocovariance_bar(p + 1);
		assert(sigma2 > 0.0);
	}
	else
	{
		assert(false);
	}

	// POSTCONDITIONS [] >>
	//	-. all variances have to be positive.
}

void ArHmmWithUnivariateNormalObservations::doEstimateObservationDensityParametersByML(const std::vector<size_t> &Ns, const unsigned int state, const std::vector<dmatrix_type> &observationSequences, const std::vector<dmatrix_type> &gammas, const size_t R, const double denominatorA)
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
	double meanWinObs;
	for (r = 0; r < R; ++r)
	{
		assert(Ns[r] > P_);

		const dmatrix_type &observationr = observationSequences[r];
		//const dmatrix_type &gammar = gammas[r];
		dmatrix_type &autocovariancer = autocovariances[r];

		// calculate autocovariance functions.
		autocovariancer.resize(Ns[r], P_ + 1, false);
		autocovariancer.clear();
		for (n = 0; n < Ns[r]; ++n)
		{
			for (w = -W; w <= W; ++w)
			{
				// TODO [check] >> which one is better?
				//winObs(w + W) = (int(n) + w < 0 || int(n) + w >= int(Ns[r])) ? 0.0 : observationr(n + w, 0);
				winObs(w + W) = int(n) + w < 0 ? observationr(0, 0) : (int(n) + w >= int(Ns[r]) ? observationr(Ns[r] - 1, 0) : observationr(n + w, 0));
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

	dvector_type autocovariance_bar(P_ + 1, 0.0);
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
	boost::numeric::ublas::matrix<double> A(P_, P_, 0.0);
	boost::numeric::ublas::vector<double> x(P_, 0.0);
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
		dvector_type &coeff = coeffs_[state];
		coeff = x;

		// reestimate the variances of the input noise process.
		double &sigma2 = sigmas_[state];
		sigma2 = autocovariance_bar(0);
		for (p = 0; p < P_; ++p)
			sigma2 -= coeff(p) * autocovariance_bar(p + 1);
		assert(sigma2 > 0.0);
	}
	else
	{
		assert(false);
	}

	// POSTCONDITIONS [] >>
	//	-. all variances have to be positive.
}

void ArHmmWithUnivariateNormalObservations::doEstimateObservationDensityParametersByMAPUsingConjugatePrior(const size_t N, const unsigned int state, const dmatrix_type &observations, const dmatrix_type &gamma, const double denominatorA)
{
	// FIXME [modify] >>
	throw std::runtime_error("Not yet implemented");

#if 0
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
#endif
}

void ArHmmWithUnivariateNormalObservations::doEstimateObservationDensityParametersByMAPUsingConjugatePrior(const std::vector<size_t> &Ns, const unsigned int state, const std::vector<dmatrix_type> &observationSequences, const std::vector<dmatrix_type> &gammas, const size_t R, const double denominatorA)
{
	// FIXME [modify] >>
	throw std::runtime_error("Not yet implemented");

#if 0
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
#endif
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

	double M = 0.0;

	//const dvector_type &coeff = coeffs_[state];
	for (size_t p = 0; p < P_; ++p)
	{
		// FIXME [check] > is it correct?
		// TODO [check] >> which one is better?
		M = 0.0;  //coeff(p) * 0.0;
		//M = coeff(p) * observation(0);
	}

	//boost::math::normal pdf;  // (default mean = zero, and standard deviation = unity).
	boost::math::normal pdf(M, sigmas_[state]);

	return boost::math::pdf(pdf, observation(0));
}

double ArHmmWithUnivariateNormalObservations::doEvaluateEmissionProbability(const unsigned int state, const size_t n, const dmatrix_type &observations) const
{
	double M = 0.0;

	const dvector_type &coeff = coeffs_[state];
	for (size_t p = 0; p < P_; ++p)
	{
		// TODO [check] >> which one is better?
		//M = coeff(p) * ((int(n) - int(p) - 1 < 0) ? 0.0 : observations(n-p-1, 0));
		M = coeff(p) * ((int(n) - int(p) - 1 < 0) ? observations(0, 0) : observations(n-p-1, 0));
	}

	//boost::math::normal pdf;  // (default mean = zero, and standard deviation = unity).
	boost::math::normal pdf(M, sigmas_[state]);

	return boost::math::pdf(pdf, observations(n, 0));
}

void ArHmmWithUnivariateNormalObservations::doGenerateObservationsSymbol(const unsigned int state, const size_t n, dmatrix_type &observations) const
{
	typedef boost::normal_distribution<> distribution_type;
	typedef boost::variate_generator<base_generator_type &, distribution_type> generator_type;

	double M = 0.0;
	for (size_t p = 0; p < P_; ++p)
	{
		// TODO [check] >> which one is better?
		//M = coeffs_[state](p) * ((int(n) - int(p) - 1 < 0) ? 0.0 : observations(n-p-1, 0));
		M = coeffs_[state](p) * ((int(n) - int(p) - 1 < 0) ? observations(0, 0) : observations(n-p-1, 0));
	}

	generator_type normal_gen(baseGenerator_, distribution_type(M, sigmas_[state]));
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

	size_t k, p;

	// K x P.
	for (k = 0; k < K_; ++k)
		for (p = 0; p < P_; ++p)
			stream >> coeffs_[k](p);

	stream >> dummy;
#if defined(__GNUC__)
	if (strcasecmp(dummy.c_str(), "sigma:") != 0)
#elif defined(_MSC_VER)
	if (_stricmp(dummy.c_str(), "sigma:") != 0)
#endif
		return false;

	// K.
	for (k = 0; k < K_; ++k)
		stream >> sigmas_[k];

	return true;
}

bool ArHmmWithUnivariateNormalObservations::doWriteObservationDensity(std::ostream &stream) const
{
	stream << "ar univariate normal:" << std::endl;

	stream << "P= " << P_ << std::endl;  // the order of autoregressive model.

	size_t k, p;

	// K x P.
	stream << "coeff:" << std::endl;
	for (k = 0; k < K_; ++k)
	{
		for (p = 0; p < P_; ++p)
			stream << coeffs_[k](p) << ' ';
		stream << std::endl;
	}
	//stream << std::endl;

	// K.
	stream << "sigma:" << std::endl;
	for (k = 0; k < K_; ++k)
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

	const std::size_t numParameters = K_ * D_ * P_ + K_ * D_;  // the total number of parameters of observation density.

	assert(numLowerBound == numUpperBound);
	assert(1 == numLowerBound || numParameters == numLowerBound);

	if (1 == numLowerBound)
	{
		const double lb = lowerBoundsOfObservationDensity[0], ub = upperBoundsOfObservationDensity[0];
		size_t k, p;
		for (k = 0; k < K_; ++k)
		{
			for (p = 0; p < P_; ++p)
				coeffs_[k](p) = ((double)std::rand() / RAND_MAX) * (ub - lb) + lb;
			// TODO [check] >> all variances have to be positive.
			sigmas_[k] = ((double)std::rand() / RAND_MAX) * (ub - lb) + lb;
		}
	}
	else if (numParameters == numLowerBound)
	{
		size_t k, p, idx = 0;
		for (k = 0; k < K_; ++k)
			for (p = 0; p < P_; ++p, ++idx)
				coeffs_[k](p) = ((double)std::rand() / RAND_MAX) * (upperBoundsOfObservationDensity[idx] - lowerBoundsOfObservationDensity[idx]) + lowerBoundsOfObservationDensity[idx];
		for (k = 0; k < K_; ++k, ++idx)
			// TODO [check] >> all variances have to be positive.
			sigmas_[k] = ((double)std::rand() / RAND_MAX) * (upperBoundsOfObservationDensity[idx] - lowerBoundsOfObservationDensity[idx]) + lowerBoundsOfObservationDensity[idx];
	}

	for (size_t k = 0; k < K_; ++k)
	{
		// all variances have to be positive.
		if (sigmas_[k] < 0.0)
			sigmas_[k] = -sigmas_[k];
	}

	// POSTCONDITIONS [] >>
	//	-. all variances have to be positive.
}

}  // namespace swl
