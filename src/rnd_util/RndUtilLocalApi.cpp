#include "swl/Config.h"
#include "RndUtilLocalApi.h"
#include "swl/math/MathUtil.h"
#include <gsl/gsl_roots.h>
#include <gsl/gsl_errno.h>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/blas.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/constants/constants.hpp>
#include <ctime>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


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

//--------------------------------------------------------------------------
// von Mises target distribution

double VonMisesTargetDistribution::evaluate(const vector_type &x) const
{
	return 0.5 * std::exp(kappa_ * std::cos(x[0] - mean_direction_)) / (boost::math::constants::pi<double>() * boost::math::cyl_bessel_i(0.0, kappa_));;
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
	sample[0] = swl::MathUtil::wrap(generator_(), 0.0, 2.0 * boost::math::constants::pi<double>());
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
	sample[0] = swl::MathUtil::wrap(((double)std::rand() / RAND_MAX) * (upper_ - lower_) + lower_, 0.0, 2.0 * boost::math::constants::pi<double>());
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

}  // namespace swl
