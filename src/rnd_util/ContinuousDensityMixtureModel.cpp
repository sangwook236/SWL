#include "swl/Config.h"
#include "swl/rnd_util/ContinuousDensityMixtureModel.h"
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <cstring>
#include <stdexcept>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

ContinuousDensityMixtureModel::ContinuousDensityMixtureModel(const std::size_t K, const std::size_t D)
: base_type(K, D)
{
}

ContinuousDensityMixtureModel::ContinuousDensityMixtureModel(const std::size_t K, const std::size_t D, const std::vector<double> &pi)
: base_type(K, D, pi)
{
}

ContinuousDensityMixtureModel::~ContinuousDensityMixtureModel()
{
}

void ContinuousDensityMixtureModel::computeGamma(const std::size_t N, const dmatrix_type &observations, dmatrix_type &gamma, double &logProbability) const
{
	std::size_t k;
	double denominator;

	logProbability = 0.0;
	for (std::size_t n = 0; n < N; ++n)
	{
		denominator = 0.0;
		for (k = 0; k < K_; ++k)
		{
			//gamma(n, k) = pi_[k] * evaluateEmissionProbability(k, boost::numeric::ublas::matrix_row<const dmatrix_type>(observations, n));  // error !!!
			gamma(n, k) = pi_[k] * doEvaluateMixtureComponentProbability(k, boost::numeric::ublas::matrix_row<const dmatrix_type>(observations, n));
			denominator += gamma(n, k);
		}

		logProbability += std::log(denominator);

		for (k = 0; k < K_; ++k)
			gamma(n, k) = gamma(n, k) / denominator;
	}
}

bool ContinuousDensityMixtureModel::estimateParametersByML(const std::size_t N, const dmatrix_type &observations, const double terminationTolerance, const std::size_t maxIteration, std::size_t &numIteration, double &initLogProbability, double &finalLogProbability)
{
	dmatrix_type gamma(N, K_, 0.0);
	double logProb;

	// E-step
	computeGamma(N, observations, gamma, logProb);

	initLogProbability = logProb;  // log P(observations | initial model)
	finalLogProbability = logProb;

	double delta;
	std::size_t k, n;
	numIteration = 0;
	double sumGamma;
	do
	{
		// M-step
		for (k = 0; k < K_; ++k)
		{
			// reestimate the mixture coefficient of state k
			sumGamma = 0.0;
			for (n = 0; n < N; ++n)
				sumGamma += gamma(n, k);
			pi_[k] = 0.001 + 0.999 * sumGamma / N;

			// reestimate observation(emission) distribution in each state
			doEstimateObservationDensityParametersByML(N, (unsigned int)k, observations, gamma, sumGamma);
		}

		// E-step
		computeGamma(N, observations, gamma, logProb);

		// compute difference between log probability of two iterations
#if 1
		delta = logProb - finalLogProbability;
#else
		delta = std::fabs(logProb - finalLogProbability);
#endif

		finalLogProbability = logProb;  // log P(observations | estimated model)
		++numIteration;
	} while (delta > terminationTolerance && numIteration <= maxIteration);  // if log probability does not change much, exit

	return true;
}

bool ContinuousDensityMixtureModel::estimateParametersByMAP(const std::size_t N, const dmatrix_type &observations, const double terminationTolerance, const std::size_t maxIteration, std::size_t &numIteration, double &initLogProbability, double &finalLogProbability)
{
	// FIXME [modify] >>
	throw std::runtime_error("not yet implemented");

	dmatrix_type gamma(N, K_, 0.0);
	double logProb;

	// E-step
	computeGamma(N, observations, gamma, logProb);

	initLogProbability = logProb;  // log P(observations | initial model)
	finalLogProbability = logProb;

	double delta;
	std::size_t n, k;
	numIteration = 0;
	double sumGamma;
	do
	{
		// M-step
		for (k = 0; k < K_; ++k)
		{
			sumGamma = 0.0;
			for (n = 0; n < N; ++n)
				sumGamma += gamma(n, k);

			// reestimate the mixture coefficient of state k
			pi_[k] = 0.001 + 0.999 * sumGamma / N;

			// reestimate observation(emission) distribution in each state
			doEstimateObservationDensityParametersByML(N, (unsigned int)k, observations, gamma, sumGamma);
		}

		// E-step
		computeGamma(N, observations, gamma, logProb);

		// compute difference between log probability of two iterations
#if 1
		delta = logProb - finalLogProbability;
#else
		delta = std::fabs(logProb - finalLogProbability);
#endif

		finalLogProbability = logProb;  // log P(observations | estimated model)
		++numIteration;
	} while (delta > terminationTolerance && numIteration <= maxIteration);  // if log probability does not change much, exit

	return true;
}

void ContinuousDensityMixtureModel::generateSample(const std::size_t N, dmatrix_type &observations, std::vector<unsigned int> &states, const unsigned int seed /*= (unsigned int)-1*/) const
{
	// PRECONDITIONS [] >>
	//	-. std::srand() had to be called before this function is called.

	for (std::size_t n = 0; n < N; ++n)
	{
		states[n] = generateState();
#if defined(__GNUC__)
		boost::numeric::ublas::matrix_row<const dmatrix_type> obs(observations, n);
		doGenerateObservationsSymbol(states[n], obs, (unsigned int)-1);
#else
		doGenerateObservationsSymbol(states[n], boost::numeric::ublas::matrix_row<dmatrix_type>(observations, n), (unsigned int)-1);
#endif
	}
}

double ContinuousDensityMixtureModel::evaluateEmissionProbability(const boost::numeric::ublas::matrix_row<const dmatrix_type> &observation) const
{
	double prob = 0.0;
	for (size_t k = 0; k < K_; ++k)
		prob += pi_[k] * doEvaluateMixtureComponentProbability(k, observation);

	return prob;
}

/*static*/ bool ContinuousDensityMixtureModel::readSequence(std::istream &stream, std::size_t &N, std::size_t &D, dmatrix_type &observations)
{
	std::string dummy;

	stream >> dummy >> N;
#if defined(__GNUC__)
	if (strcasecmp(dummy.c_str(), "N=") != 0)
#elif defined(_MSC_VER)
	if (_stricmp(dummy.c_str(), "N=") != 0)
#endif
		return false;

	stream >> dummy >> D;
#if defined(__GNUC__)
	if (strcasecmp(dummy.c_str(), "D=") != 0)
#elif defined(_MSC_VER)
	if (_stricmp(dummy.c_str(), "D=") != 0)
#endif
		return false;

	observations.resize(N, D);
	for (std::size_t n = 0; n < N; ++n)
		for (std::size_t i = 0; i < D; ++i)
			stream >> observations(n, i);

	return true;
}

/*static*/ bool ContinuousDensityMixtureModel::writeSequence(std::ostream &stream, const dmatrix_type &observations)
{
	const std::size_t N = observations.size1();
	const std::size_t D = observations.size2();

	stream << "N= " << N << std::endl;
	stream << "D= " << D << std::endl;
	for (std::size_t n = 0; n < N; ++n)
	{
		for (std::size_t i = 0; i < D; ++i)
			stream << observations(n, i) << ' ';
		stream << std::endl;
	}

	return true;
}

}  // namespace swl
