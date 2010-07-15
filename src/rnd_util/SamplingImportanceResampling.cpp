#include "swl/Config.h"
#include "swl/rnd_util/SamplingImportanceResampling.h"
#include <boost/random/linear_congruential.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <stdexcept>
#include <ctime>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//--------------------------------------------------------------------------
// sampling importance resampling (SIR): sequential importance sampling (SIS) + resampling
// particle filter

// [ref]
// "On sequential Monte Carlo sampling methods for Bayesian filtering", Arnaud Doucet, Simon Godsill, and Christophe Andrieu,
//	Statistics and Computing, 10, pp. 197-208, 2000
// "An Introduction to Sequential Monte Carlo Methods", Arnaud Doucet, Nando de Freitas, and Neil Gordon, 2001
// "Kalman Filtering and Neural Networks", Simon Haykin, 2001, Ch. 7

SamplingImportanceResampling::SamplingImportanceResampling(const double effectiveSampleSize, TransitionDistribution &transitionDistribution, ObservationDistribution &observationDistribution, ProposalDistribution &proposalDistribution)
: effectiveSampleSize_(effectiveSampleSize),
  transitionDistribution_(transitionDistribution), observationDistribution_(observationDistribution), proposalDistribution_(proposalDistribution),
  baseGenerator_(static_cast<unsigned int>(std::time(NULL))), generator_(baseGenerator_, boost::uniform_real<>(0, 1))
{
	//baseGenerator_.seed(static_cast<unsigned int>(std::time(NULL)));
}

SamplingImportanceResampling::~SamplingImportanceResampling()
{
}

void SamplingImportanceResampling::sample(const size_t step, const size_t particleNum, const std::vector<vector_type> &xs, const vector_type &y, std::vector<vector_type> &newXs, std::vector<double> &weights, vector_type *estimatedX /*= NULL*/) const
{
	const double eps = 1.0e-15;

	//-----------------------------------------------------
	// 1. importance sampling

	double weightSum = 0.0;
	for (size_t i = 0; i < particleNum; ++i)
	{
		proposalDistribution_.sample(step, xs[i], y, newXs[i]);

		const double obsvProb = observationDistribution_.evaluate(step, xs[i], y);
		const double tranProb = transitionDistribution_.evaluate(step, newXs[i], xs[i]);
		const double phi = proposalDistribution_.evaluate(step, newXs[i], xs[i], y);

		if (phi > eps)
			weights[i] *= obsvProb * tranProb / phi;
		else throw std::runtime_error("divide by zero");

		weightSum += weights[i];
	}

	// normalize the importance weights
	double Neff = 0.0;
	for (std::vector<double>::iterator it = weights.begin(); it != weights.end(); ++it)
	{
		*it /= weightSum;
		Neff += (*it) * (*it);
	}
	Neff = 1.0 / Neff;

	// find a particle with a maximum weight
	if (estimatedX)
	{
		std::vector<double>::iterator maxIt = std::max_element(weights.begin(), weights.end());
		const size_t maxIdx = (size_t)std::distance(weights.begin(), maxIt);
		*estimatedX = newXs[maxIdx];
	}

	//-----------------------------------------------------
	// 2. resampling

	if (Neff > effectiveSampleSize_)  // sequential importance sampling (SIS)
	{
		// do nothing
	}
	else  // sampling importance resampling (SIR)
	{
		// FIXME [modify] >>

		std::vector<double> cdf(particleNum);
		cdf[0] = weights[0];
		for (size_t i = 1; i < particleNum; ++i)
			cdf[i] = cdf[i - 1] + weights[i];

		std::vector<vector_type> resampledXs;
		resampledXs.reserve(particleNum);
		for (size_t i = 0; i < particleNum; ++i)
		{
			const double &u = generator_();
			std::vector<double>::iterator up = std::upper_bound(cdf.begin(), cdf.end(), u);
			const size_t &idx = (size_t)std::distance(cdf.begin(), up);
			resampledXs.push_back(newXs[idx]);
		}

		newXs.swap(resampledXs);
		weights.assign(particleNum, 1.0 / particleNum);

		// MCMC move step (optional)
		// "Kalman Filtering and Neural Networks", Simon Haykin, 2001, Ch. 7
		// "An Introduction to MCMC for Machine Learning", Christophe Andrieu, Nando de Freitas, Arnaud Doucet, and Michael I. Jordan
		//	Machine Learning, 50, pp. 5-43, 2003
	}
}

}  // namespace swl
