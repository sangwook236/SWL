#include "swl/Config.h"
#include "swl/rnd_util/HmmWithVonMisesObservations.h"
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

namespace {

double kappa_objective_function(double x, void *params)
{
	double *A = (double *)params;

	return boost::math::cyl_bessel_i(1.0, x) / boost::math::cyl_bessel_i(0.0, x) - *A;
}

bool one_dim_root_finding_using_f(const double A, const double upper, double &kappa)
{
	gsl_function func;
	func.function = &kappa_objective_function;
	func.params = (void *)&A;

	//const gsl_root_fsolver_type *T = gsl_root_fsolver_bisection;
	//const gsl_root_fsolver_type *T = gsl_root_fsolver_falsepos;
	const gsl_root_fsolver_type *T = gsl_root_fsolver_brent;
	gsl_root_fsolver *s = gsl_root_fsolver_alloc(T);

	double x_lo = 0.0, x_hi = upper;
	gsl_root_fsolver_set(s, &func, x_lo, x_hi);

	//std::cout << "===== using " << gsl_root_fsolver_name(s) << " method =====" << std::endl;
	//std::cout << std::setw(5) << "iter" << " [" << std::setw(9) << "lower" << ", " << std::setw(9) << "upper" << "] " << std::setw(9) << "root" << std::setw(10) << "err(est)" << std::endl;

	int status;
	int iter = 0, max_iter = 100;
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
	} while (GSL_CONTINUE == status && iter < max_iter);

	if (GSL_SUCCESS != status)
	{
		//std::cout << "not converged" << std::endl;
		kappa = 0.0;
		return false;
	}

	return true;
}

}  // unnamed namespace

HmmWithVonMisesObservations::HmmWithVonMisesObservations(const size_t K)
: base_type(K, 1), mus_(K, 0.0), kappas_(K, 0.0)  // 0-based index
{
}

HmmWithVonMisesObservations::HmmWithVonMisesObservations(const size_t K, const dvector_type &pi, const dmatrix_type &A, const dvector_type &mus, const dvector_type &kappas)
: base_type(K, 1, pi, A), mus_(mus), kappas_(kappas)
{
}

HmmWithVonMisesObservations::~HmmWithVonMisesObservations()
{
}

void HmmWithVonMisesObservations::doEstimateObservationDensityParametersInMStep(const size_t N, const unsigned int state, const dmatrix_type &observations, dmatrix_type &gamma, const double denominatorA)
{
	// reestimate symbol prob in each state

	size_t n;
	double numeratorPr = 0.0, denominatorPr = 0.0;
	for (n = 0; n < N; ++n)
	{
		numeratorPr += gamma(n, state) * std::sin(observations(n, 0));
		denominatorPr += gamma(n, state) * std::cos(observations(n, 0));
	}

	// TODO [check] >> check the range of each mu, [0, 2 * pi)
	//mus_[state] = std::atan2(numeratorPr, denominatorPr);
	mus_[state] = std::atan2(numeratorPr, denominatorPr) + boost::math::constants::pi<double>();
	//mus_[state] = 0.001 + 0.999 * std::atan2(numeratorPr, denominatorPr) + boost::math::constants::pi<double>();

	//
	denominatorPr = denominatorA + gamma(N-1, state);
	numeratorPr = 0.0;
	for (n = 0; n < N; ++n)
		numeratorPr += gamma(n, state) * std::cos(observations(n, 0) - mus_[state]);

	const double A = 0.001 + 0.999 * numeratorPr / denominatorPr;
	// FIXME [modify] >> upper bound has to be adjusted
	const double ub = 10000.0;  // kappa >= 0.0
	const bool retval = one_dim_root_finding_using_f(A, ub, kappas_[state]);
	assert(retval);
}

void HmmWithVonMisesObservations::doEstimateObservationDensityParametersInMStep(const std::vector<size_t> &Ns, const unsigned int state, const std::vector<dmatrix_type> &observationSequences, const std::vector<dmatrix_type> &gammas, const size_t R, const double denominatorA)
{
	// reestimate symbol prob in each state

	size_t n, r;
	double numeratorPr = 0.0, denominatorPr = 0.0;
	for (r = 0; r < R; ++r)
	{
		const dmatrix_type &observationr = observationSequences[r];
		const dmatrix_type &gammar = gammas[r];

		for (n = 0; n < Ns[r]; ++n)
		{
			numeratorPr += gammar(n, state) * std::sin(observationr(n, 0));
			denominatorPr += gammar(n, state) * std::cos(observationr(n, 0));
		}
	}

	// TODO [check] >> check the range of each mu, [0, 2 * pi)
	//mus_[state] = std::atan2(numeratorPr, denominatorPr);
	mus_[state] = std::atan2(numeratorPr, denominatorPr) + boost::math::constants::pi<double>();
	//mus_[state] = 0.001 + 0.999 * std::atan2(numeratorPr, denominatorPr) + boost::math::constants::pi<double>();

	//
	denominatorPr = denominatorA;
	for (r = 0; r < R; ++r)
		denominatorPr += gammas[r](Ns[r]-1, state);

	numeratorPr = 0.0;
	for (r = 0; r < R; ++r)
	{
		const dmatrix_type &observationr = observationSequences[r];
		const dmatrix_type &gammar = gammas[r];

		for (n = 0; n < Ns[r]; ++n)
			numeratorPr += gammar(n, state) * std::cos(observationr(n, 0) - mus_[state]);
	}

	const double A = 0.001 + 0.999 * numeratorPr / denominatorPr;
	// FIXME [modify] >> upper bound has to be adjusted
	const double ub = 10000.0;  // kappa >= 0.0
	const bool retval = one_dim_root_finding_using_f(A, ub, kappas_[state]);
	assert(retval);
}

double HmmWithVonMisesObservations::doEvaluateEmissionProbability(const unsigned int state, const boost::numeric::ublas::matrix_row<const dmatrix_type> &observation) const
{
	// each observation are expressed as a random angle, 0 <= observation[0] < 2 * pi. [rad].
	return 0.5 * std::exp(kappas_[state] * std::cos(observation[0] - mus_[state])) / (boost::math::constants::pi<double>() * boost::math::cyl_bessel_i(0.0, kappas_[state]));
}

void HmmWithVonMisesObservations::doGenerateObservationsSymbol(const unsigned int state, boost::numeric::ublas::matrix_row<dmatrix_type> &observation, const unsigned int seed /*= (unsigned int)-1*/) const
{
	throw std::runtime_error("not yet implemented");
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

	for (size_t k = 0; k < K_; ++k)
		stream >> mus_[k];

	stream >> dummy;
#if defined(__GNUC__)
	if (strcasecmp(dummy.c_str(), "kappa:") != 0)
#elif defined(_MSC_VER)
	if (_stricmp(dummy.c_str(), "kappa:") != 0)
#endif
		return false;

	for (size_t k = 0; k < K_; ++k)
		stream >> kappas_[k];

	return true;
}

bool HmmWithVonMisesObservations::doWriteObservationDensity(std::ostream &stream) const
{
	stream << "von Mises:" << std::endl;

	stream << "mu:" << std::endl;
	for (size_t k = 0; k < K_; ++k)
		stream << mus_[k] << ' ';
	stream << std::endl;

	stream << "kappa:" << std::endl;
	for (size_t k = 0; k < K_; ++k)
		stream << kappas_[k] << ' ';
	stream << std::endl;

	return true;
}

void HmmWithVonMisesObservations::doInitializeObservationDensity()
{
	// PRECONDITIONS [] >>
	//	-. std::srand() had to be called before this function is called.

	// FIXME [modify] >> lower & upper bounds have to be adjusted
	const double lb = -10000.0, ub = 10000.0;
	for (size_t k = 0; k < K_; ++k)
	{
		mus_[k] = ((double)std::rand() / RAND_MAX) * (ub - lb) + lb;
		kappas_[k] = ((double)std::rand() / RAND_MAX) * (ub - lb) + lb;
	}
}

}  // namespace swl
