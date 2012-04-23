#include "swl/Config.h"
#include "swl/rnd_util/HmmWithUnivariateNormalMixtureObservations.h"
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/math/distributions/normal.hpp>  // for normal distribution
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stdexcept>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

HmmWithUnivariateNormalMixtureObservations::HmmWithUnivariateNormalMixtureObservations(const size_t K, const size_t C)
: base_type(K, 1), HmmWithMixtureObservations(C, K), mus_(K, C, 0.0), sigmas_(K, C, 0.0)  // 0-based index
{
}

HmmWithUnivariateNormalMixtureObservations::HmmWithUnivariateNormalMixtureObservations(const size_t K, const size_t C, const dvector_type &pi, const dmatrix_type &A, const dmatrix_type &alphas, const dmatrix_type &mus, const dmatrix_type &sigmas)
: base_type(K, 1, pi, A), HmmWithMixtureObservations(C, K, alphas), mus_(mus), sigmas_(sigmas)
{
}

HmmWithUnivariateNormalMixtureObservations::~HmmWithUnivariateNormalMixtureObservations()
{
}

void HmmWithUnivariateNormalMixtureObservations::doEstimateObservationDensityParametersInMStep(const size_t N, const unsigned int state, const dmatrix_type &observations, dmatrix_type &gamma, const double denominatorA)
{
	// reestimate symbol prob in each state

	size_t c, n;
	double numerator, denominator;

	// E-step
	// TODO [check] >> frequent memory reallocation may make trouble
	dmatrix_type zeta(N, C_, 0.0);
	{
		std::vector<boost::math::normal> pdfs;
		pdfs.reserve(C_);
		for (c = 0; c < C_; ++c)
			pdfs.push_back(boost::math::normal(mus_(state, c), sigmas_(state, c)));

		double val;
		for (n = 0; n < N; ++n)
		{
			const boost::numeric::ublas::matrix_row<const dmatrix_type> obs(observations, n);

			denominator = 0.0;
			for (c = 0; c < C_; ++c)
			{
				// FIXME [fix] >>
				//val = alphas_(state, c) * doEvaluateEmissionProbability(state, obs);  // error !!!
				val = alphas_(state, c) * boost::math::pdf(pdfs[c], obs[0]);

				zeta(n, c) = val;
				denominator += val;
			}

			val = 0.999 * gamma(n, state) / denominator;
			for (c = 0; c < C_; ++c)
				zeta(n, c) = 0.001 + val * zeta(n, c);
		}
	}

	// M-step
	denominator = denominatorA + gamma(N-1, state);
	double sumZeta;
	for (c = 0; c < C_; ++c)
	{
		sumZeta = 0.0;
		for (n = 0; n < N; ++n)
			sumZeta += zeta(n, c);
		alphas_(state, c) = 0.001 + 0.999 * sumZeta / denominator;

		//
		numerator = 0.0;
		for (n = 0; n < N; ++n)
			numerator += zeta(n, c) * observations(n, 0);
		mus_(state, c) = 0.001 + 0.999 * numerator / sumZeta;

		//
		numerator = 0.0;
		for (n = 0; n < N; ++n)
			numerator += zeta(n, c) * (observations(n, 0) - mus_(state, c)) * (observations(n, 0) - mus_(state, c));
		sigmas_(state, c) = 0.001 + 0.999 * numerator / sumZeta;
		assert(sigmas_(state, c) > 0.0);
	}
}

void HmmWithUnivariateNormalMixtureObservations::doEstimateObservationDensityParametersInMStep(const std::vector<size_t> &Ns, const unsigned int state, const std::vector<dmatrix_type> &observationSequences, const std::vector<dmatrix_type> &gammas, const size_t R, const double denominatorA)
{
	// reestimate symbol prob in each state

	size_t c, n, r;
	double numerator, denominator;

	// E-step
	// TODO [check] >> frequent memory reallocation may make trouble
	std::vector<dmatrix_type> zetas;
	zetas.reserve(R);
	for (r = 0; r < R; ++r)
		zetas.push_back(dmatrix_type(Ns[r], C_, 0.0));

	{
		std::vector<boost::math::normal> pdfs;
		pdfs.reserve(C_);
		for (c = 0; c < C_; ++c)
			pdfs.push_back(boost::math::normal(mus_(state, c), sigmas_(state, c)));

		double val;
		for (r = 0; r < R; ++r)
			for (n = 0; n < Ns[r]; ++n)
			{
				const boost::numeric::ublas::matrix_row<const dmatrix_type> obs(observationSequences[r], n);

				denominator = 0.0;
				for (c = 0; c < C_; ++c)
				{
					// FIXME [fix] >>
					//val = alphas_(state, c) * doEvaluateEmissionProbability(state, obs);  // error !!!
					val = alphas_(state, c) * boost::math::pdf(pdfs[c], obs[0]);

					zetas[r](n, c) = val;
					denominator += val;
				}

				val = 0.999 * gammas[r](n, state) / denominator;
				for (c = 0; c < C_; ++c)
					zetas[r](n, c) = 0.001 + val * zetas[r](n, c);
			}
	}

	// M-step
	denominator = denominatorA;
	for (r = 0; r < R; ++r)
		denominator += gammas[r](Ns[r]-1, state);

	double sumZeta;
	const double factor = 0.999 / denominator;
	for (c = 0; c < C_; ++c)
	{
		sumZeta = 0.0;
		for (r = 0; r < R; ++r)
			for (n = 0; n < Ns[r]; ++n)
				sumZeta += zetas[r](n, c);
		alphas_(state, c) = 0.001 + factor * sumZeta;

		//
		numerator = 0.0;
		for (r = 0; r < R; ++r)
			for (n = 0; n < Ns[r]; ++n)
				numerator += zetas[r](n, c) * observationSequences[r](n, 0);
		mus_(state, c) = 0.001 + 0.999 * numerator / sumZeta;

		//
		numerator = 0.0;
		for (r = 0; r < R; ++r)
			for (n = 0; n < Ns[r]; ++n)
				numerator += zetas[r](n, c) * (observationSequences[r](n, 0) - mus_(state, c)) * (observationSequences[r](n, 0) - mus_(state, c));
		sigmas_(state, c) = 0.001 + 0.999 * numerator / sumZeta;
	}
}

double HmmWithUnivariateNormalMixtureObservations::doEvaluateEmissionProbability(const unsigned int state, const boost::numeric::ublas::matrix_row<const dmatrix_type> &observation) const
{
	double sum = 0.0;
	for (size_t c = 0; c < C_; ++c)
	{
		//boost::math::normal pdf;  // (default mean = zero, and standard deviation = unity)
		boost::math::normal pdf(mus_(state, c), sigmas_(state, c));

		sum += alphas_(state, c) * boost::math::pdf(pdf, observation[0]);
	}

	return sum;
}

void HmmWithUnivariateNormalMixtureObservations::doGenerateObservationsSymbol(const unsigned int state, boost::numeric::ublas::matrix_row<dmatrix_type> &observation, const unsigned int seed /*= (unsigned int)-1*/) const
{
	if ((unsigned int)-1 != seed)
		baseGenerator_.seed(seed);

	const double prob = (double)std::rand() / RAND_MAX;

	double accum = 0.0;
	unsigned int component = (unsigned int)C_;
	for (size_t c = 0; c < C_; ++c)
	{
		accum += alphas_(state, c);
		if (prob < accum)
		{
			component = (unsigned int)c;
			break;
		}
	}

	// TODO [check] >>
	if ((unsigned int)C_ == component)
		component = (unsigned int)(C_ - 1);

	//
	typedef boost::normal_distribution<> distribution_type;
	typedef boost::variate_generator<base_generator_type &, distribution_type> generator_type;

	generator_type normal_gen(baseGenerator_, distribution_type(mus_(state, component), sigmas_(state, component)));
	for (size_t d = 0; d < D_; ++d)
		observation[d] = normal_gen();
}

bool HmmWithUnivariateNormalMixtureObservations::doReadObservationDensity(std::istream &stream)
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
	if (strcasecmp(dummy.c_str(), "normal") != 0)
#elif defined(_MSC_VER)
	if (_stricmp(dummy.c_str(), "normal") != 0)
#endif
		return false;

	stream >> dummy;
#if defined(__GNUC__)
	if (strcasecmp(dummy.c_str(), "mixture:") != 0)
#elif defined(_MSC_VER)
	if (_stricmp(dummy.c_str(), "mixture:") != 0)
#endif
		return false;

	// TODO [check] >>
	size_t C;
	stream >> dummy >> C;  // the number of mixture components
#if defined(__GNUC__)
	if (strcasecmp(dummy.c_str(), "C=") != 0 || C_ != C)
#elif defined(_MSC_VER)
	if (_stricmp(dummy.c_str(), "C=") != 0 || C_ != C)
#endif
		return false;

	stream >> dummy;
#if defined(__GNUC__)
	if (strcasecmp(dummy.c_str(), "alpha:") != 0)
#elif defined(_MSC_VER)
	if (_stricmp(dummy.c_str(), "alpha:") != 0)
#endif
		return false;

	for (size_t k = 0; k < K_; ++k)
		for (size_t c = 0; c < C_; ++c)
			stream >> alphas_(k, c);

	stream >> dummy;
#if defined(__GNUC__)
	if (strcasecmp(dummy.c_str(), "mu:") != 0)
#elif defined(_MSC_VER)
	if (_stricmp(dummy.c_str(), "mu:") != 0)
#endif
		return false;

	for (size_t k = 0; k < K_; ++k)
		for (size_t c = 0; c < C_; ++c)
			stream >> mus_(k, c);

	stream >> dummy;
#if defined(__GNUC__)
	if (strcasecmp(dummy.c_str(), "sigma:") != 0)
#elif defined(_MSC_VER)
	if (_stricmp(dummy.c_str(), "sigma:") != 0)
#endif
		return false;

	for (size_t k = 0; k < K_; ++k)
		for (size_t c = 0; c < C_; ++c)
			stream >> sigmas_(k, c);

	return true;
}

bool HmmWithUnivariateNormalMixtureObservations::doWriteObservationDensity(std::ostream &stream) const
{
	stream << "univariate normal mixture:" << std::endl;

	stream << "C= " << C_ << std::endl;  // the number of mixture components

	stream << "alpha:" << std::endl;
	for (size_t k = 0; k < K_; ++k)
	{
		for (size_t c = 0; c < C_; ++c)
			stream << alphas_(k, c) << ' ';
		stream << std::endl;
	}

	stream << "mu:" << std::endl;
	for (size_t k = 0; k < K_; ++k)
	{
		for (size_t c = 0; c < C_; ++c)
			stream << mus_(k, c) << ' ';
		stream << std::endl;
	}

	stream << "sigma:" << std::endl;
	for (size_t k = 0; k < K_; ++k)
	{
		for (size_t c = 0; c < C_; ++c)
			stream << sigmas_(k, c) << ' ';
		stream << std::endl;
	}

	return true;
}

void HmmWithUnivariateNormalMixtureObservations::doInitializeObservationDensity()
{
	// PRECONDITIONS [] >>
	//	-. std::srand() had to be called before this function is called.

	// FIXME [modify] >> lower & upper bounds have to be adjusted
	const double lb = -10000.0, ub = 10000.0;
	double sum;
	size_t c;
	for (size_t k = 0; k < K_; ++k)
	{
		sum = 0.0;
		for (c = 0; c < C_; ++c)
		{
			alphas_(k, c) = (double)std::rand() / RAND_MAX;
			sum += alphas_(k, c);

			mus_(k, c) = ((double)std::rand() / RAND_MAX) * (ub - lb) + lb;
			sigmas_(k, c) = ((double)std::rand() / RAND_MAX) * (ub - lb) + lb;
		}
		for (c = 0; c < C_; ++c)
			alphas_(k, c) /= sum;
	}
}

}  // namespace swl
