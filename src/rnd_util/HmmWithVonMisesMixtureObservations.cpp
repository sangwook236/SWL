#include "swl/Config.h"
#include "swl/rnd_util/HmmWithVonMisesMixtureObservations.h"
#include <gsl/gsl_roots.h>
#include <gsl/gsl_errno.h>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/bessel.hpp>
#include <stdexcept>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

HmmWithVonMisesMixtureObservations::HmmWithVonMisesMixtureObservations(const size_t K, const size_t C)
: base_type(K, 1), HmmWithMixtureObservations(C, K), mus_(K, C), kappas_(K, C)  // 0-based index
{
}

HmmWithVonMisesMixtureObservations::HmmWithVonMisesMixtureObservations(const size_t K, const size_t C, const dvector_type &pi, const dmatrix_type &A, const dmatrix_type &alphas, const dmatrix_type &mus, const dmatrix_type &kappas)
: base_type(K, 1, pi, A), HmmWithMixtureObservations(C, K, alphas), mus_(mus), kappas_(kappas)
{
}

HmmWithVonMisesMixtureObservations::~HmmWithVonMisesMixtureObservations()
{
}

void HmmWithVonMisesMixtureObservations::doEstimateObservationDensityParametersInMStep(const size_t N, const unsigned int state, const dmatrix_type &observations, dmatrix_type &gamma, const double denominatorA)
{
	throw std::runtime_error("not yet implemented");
}

void HmmWithVonMisesMixtureObservations::doEstimateObservationDensityParametersInMStep(const std::vector<size_t> &Ns, const unsigned int state, const std::vector<dmatrix_type> &observationSequences, const std::vector<dmatrix_type> &gammas, const size_t R, const double denominatorA)
{
	throw std::runtime_error("not yet implemented");
}

double HmmWithVonMisesMixtureObservations::doEvaluateEmissionProbability(const unsigned int state, const boost::numeric::ublas::matrix_row<const dmatrix_type> &observation) const
{
	double sum = 0.0;
	for (size_t c = 0; c < C_; ++c)
	{
		// each observation are expressed as a random angle, 0 <= observation[0] < 2 * pi. [rad].
		sum += alphas_(state, c) * 0.5 * std::exp(kappas_(state, c) * std::cos(observation[0] - mus_(state, c))) / (boost::math::constants::pi<double>() * boost::math::cyl_bessel_i(0.0, kappas_(state, c)));
	}

	return sum;
}

void HmmWithVonMisesMixtureObservations::doGenerateObservationsSymbol(const unsigned int state, boost::numeric::ublas::matrix_row<dmatrix_type> &observation, const unsigned int seed /*= (unsigned int)-1*/) const
{
	throw std::runtime_error("not yet implemented");
}

bool HmmWithVonMisesMixtureObservations::doReadObservationDensity(std::istream &stream)
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
	if (strcasecmp(dummy.c_str(), "kappa:") != 0)
#elif defined(_MSC_VER)
	if (_stricmp(dummy.c_str(), "kappa:") != 0)
#endif
		return false;

	for (size_t k = 0; k < K_; ++k)
		for (size_t c = 0; c < C_; ++c)
			stream >> kappas_(k, c);

	return true;
}

bool HmmWithVonMisesMixtureObservations::doWriteObservationDensity(std::ostream &stream) const
{
	stream << "von Mises mixture:" << std::endl;

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

	stream << "kappa:" << std::endl;
	for (size_t k = 0; k < K_; ++k)
	{
		for (size_t c = 0; c < C_; ++c)
			stream << kappas_(k, c) << ' ';
		stream << std::endl;
	}

	return true;
}

void HmmWithVonMisesMixtureObservations::doInitializeObservationDensity()
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
			kappas_(k, c) = ((double)std::rand() / RAND_MAX) * (ub - lb) + lb;
		}
		for (c = 0; c < C_; ++c)
			alphas_(k, c) /= sum;
	}
}

}  // namespace swl
