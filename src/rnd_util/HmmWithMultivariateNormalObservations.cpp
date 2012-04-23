#include "swl/Config.h"
#include "swl/rnd_util/HmmWithMultivariateNormalObservations.h"
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/blas.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/math/constants/constants.hpp>
#include <stdexcept>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

namespace {

// FIXME [move] >> this positition is temporary
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

// FIXME [move] >> this positition is temporary
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

}  // unnamed namespace

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
	// reestimate symbol prob in each state

	size_t d, n;
	const double denominatorPr = denominatorA + gamma(N-1, state);

	// TODO [check] >> this code may be changed into a vector form.
	double numeratorPr;
	for (d = 0; d < D_; ++d)
	{
		numeratorPr = 0.0;
		for (n = 0; n < N; ++n)
			numeratorPr += gamma(n, state) * observations(n, d);
		mus_[state][d] = 0.001 + 0.999 * numeratorPr / denominatorPr;
	}

	//
	dmatrix_type &sigma = sigmas_[state];
	sigma.clear();
	for (n = 0; n < N; ++n)
		boost::numeric::ublas::blas_2::sr(sigma, gamma(n, state), boost::numeric::ublas::matrix_row<const dmatrix_type>(observations, n) - mus_[state]);
	sigma = sigma * (0.999 / denominatorPr) + boost::numeric::ublas::scalar_matrix<double>(sigma.size1(), sigma.size2(), 0.001);
}

void HmmWithMultivariateNormalObservations::doEstimateObservationDensityParametersInMStep(const std::vector<size_t> &Ns, const unsigned int state, const std::vector<dmatrix_type> &observationSequences, const std::vector<dmatrix_type> &gammas, const size_t R, const double denominatorA)
{
	// reestimate symbol prob in each state

	size_t d, n, r;
	double denominatorPr = denominatorA;
	for (r = 0; r < R; ++r)
		denominatorPr += gammas[r](Ns[r]-1, state);

	// TODO [check] >> this code may be changed into a vector form.
	double numeratorPr;
	const double factor = 0.999 / denominatorPr;
	for (d = 0; d < D_; ++d)
	{
		numeratorPr = 0.0;
		for (r = 0; r < R; ++r)
		{
			const dmatrix_type &gammar = gammas[r];
			const dmatrix_type &observationr = observationSequences[r];

			for (n = 0; n < Ns[r]; ++n)
				numeratorPr += gammar(n, state) * observationr(n, d);
		}
		mus_[state][d] = 0.001 + factor * numeratorPr;
	}

	//
	dmatrix_type &sigma = sigmas_[state];
	sigma.clear();
	const boost::numeric::ublas::scalar_matrix<double> sm(sigma.size1(), sigma.size2(), 0.001);
	for (r = 0; r < R; ++r)
	{
		const dmatrix_type &gammar = gammas[r];
		const dmatrix_type &observationr = observationSequences[r];

		numeratorPr = 0.0;
		for (n = 0; n < Ns[r]; ++n)
			boost::numeric::ublas::blas_2::sr(sigma, gammar(n, state), boost::numeric::ublas::matrix_row<const dmatrix_type>(observationr, n) - mus_[state]);
		sigma = sigma * factor + sm;
	}
}

double HmmWithMultivariateNormalObservations::doEvaluateEmissionProbability(const unsigned int state, const boost::numeric::ublas::matrix_row<const dmatrix_type> &observation) const
{
	const dmatrix_type &sigma = sigmas_[state];
	const double det = determinant_by_lu(sigma);
	dmatrix_type inv(sigma.size1(), sigma.size2());
	inverse_by_lu(sigma, inv);

	const dvector_type x_mu(observation - mus_[state]);
	return std::exp(-0.5 * boost::numeric::ublas::inner_prod(x_mu, boost::numeric::ublas::prod(inv, x_mu))) / std::sqrt(std::pow(2.0 * boost::math::constants::pi<double>(), (double)D_) * det);
}

void HmmWithMultivariateNormalObservations::doGenerateObservationsSymbol(const unsigned int state, boost::numeric::ublas::matrix_row<dmatrix_type> &observation, const unsigned int seed /*= (unsigned int)-1*/) const
{
	throw std::runtime_error("not yet implemented");
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

	size_t d, i;
	for (size_t k = 0; k < K_; ++k)
	{
		stream >> dummy;
#if defined(__GNUC__)
		if (strcasecmp(dummy.c_str(), "mu:") != 0)
#elif defined(_MSC_VER)
		if (_stricmp(dummy.c_str(), "mu:") != 0)
#endif
			return false;

		for (d = 0; d < D_; ++d)
			stream >> mus_[k][d];

		stream >> dummy;
#if defined(__GNUC__)
		if (strcasecmp(dummy.c_str(), "sigma:") != 0)
#elif defined(_MSC_VER)
		if (_stricmp(dummy.c_str(), "sigma:") != 0)
#endif
			return false;

		for (d = 0; d < D_; ++d)
			for (i = 0; i < D_; ++i)
				stream >> sigmas_[k](d, i);
	}

	return true;
}

bool HmmWithMultivariateNormalObservations::doWriteObservationDensity(std::ostream &stream) const
{
	stream << "multivariate normal:" << std::endl;

	size_t d, i;
	for (size_t k = 0; k < K_; ++k)
	{
		stream << "mu:" << std::endl;
		for (d = 0; d < D_; ++d)
			stream << mus_[k][d] << ' ';
		stream << std::endl;

		stream << "sigma:" << std::endl;
		for (d = 0; d < D_; ++d)
		{
			for (i = 0; i < D_; ++i)
				stream << sigmas_[k](d, i) << ' ';
			stream << std::endl;
		}
	}

	return true;
}

void HmmWithMultivariateNormalObservations::doInitializeObservationDensity()
{
	// PRECONDITIONS [] >>
	//	-. std::srand() had to be called before this function is called.

	// FIXME [modify] >> lower & upper bounds have to be adjusted
	const double lb = -10000.0, ub = 10000.0;
	size_t d, i;
	for (size_t k = 0; k < K_; ++k)
	{
		for (d = 0; d < D_; ++d)
		{
			mus_[k][d] = ((double)std::rand() / RAND_MAX) * (ub - lb) + lb;
			// FIXME [correct] >> covariance matrices must be positive definite
			for (i = 0; i < D_; ++i)
				sigmas_[k](d, i) = ((double)std::rand() / RAND_MAX) * (ub - lb) + lb;
		}
	}
}

}  // namespace swl
