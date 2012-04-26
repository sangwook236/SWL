//#include "stdafx.h"
#include "swl/Config.h"
#include "swl/rnd_util/MetropolisHastingsAlgorithm.h"
#include <boost/random/linear_congruential.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/math/distributions/normal.hpp>
#include <iostream>
#include <vector>
#include <cmath>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace {
namespace local {

// [ref]
// "An Introduction to MCMC for Machine Learning", Christophe Andrieu, Nando de Freitas, Arnaud Doucet, and Michael I. Jordan
//	Machine Learning, 50, pp. 5-43, 2003

struct TargetDistribution: public swl::MetropolisHastingsAlgorithm::TargetDistribution
{
	typedef swl::MetropolisHastingsAlgorithm::TargetDistribution base_type;
	typedef base_type::vector_type vector_type;

	/*virtual*/ double evaluate(const vector_type &x, const vector_type *param = NULL) const
	{
		const double &v = x[0];
		return 0.3 * std::exp(-0.2 * v * v) + 0.7 * std::exp(-0.2 * (v - 10.0) * (v - 10.0));
	}
};

struct UnivariateNormalProposalDistribution: public swl::MetropolisHastingsAlgorithm::ProposalDistribution
{
	typedef swl::MetropolisHastingsAlgorithm::ProposalDistribution base_type;
	typedef base_type::vector_type vector_type;

	UnivariateNormalProposalDistribution(const double sigma)
	: base_type(), sigma_(sigma), baseGenerator_(static_cast<unsigned int>(std::time(NULL))), generator_(baseGenerator_, boost::normal_distribution<>(0.0, sigma_))
	{}

	/*virtual*/ double evaluate(const vector_type &x, const vector_type &param) const
	{
		const double &mean = param[0];
		boost::math::normal dist(mean, sigma_);
		return boost::math::pdf(dist, x[0]);
	}
	/*virtual*/ void sample(const vector_type &param, vector_type &sample) const
	{
		const double &mean = param[0];
		sample[0] = mean + generator_();
	}

private:
	typedef boost::minstd_rand base_generator_type;
	typedef boost::variate_generator<base_generator_type &, boost::normal_distribution<> > generator_type;

private:
	const double sigma_;

	base_generator_type baseGenerator_;
	mutable generator_type generator_;
};

}  // namespace local
}  // unnamed namespace

void metropolis_hastings_algorithm()
{
	const std::size_t STATE_DIM = 1;
	const double sigma = 10.0;

	local::TargetDistribution targetDist;
	local::UnivariateNormalProposalDistribution proposalDist(sigma);
	swl::MetropolisHastingsAlgorithm sampler(targetDist, proposalDist);

	swl::MetropolisHastingsAlgorithm::vector_type x(STATE_DIM, 0.0);
	swl::MetropolisHastingsAlgorithm::vector_type newX(STATE_DIM, 0.0);

	//
	const std::size_t numSample = 10000;

	std::vector<double> samples;
	samples.reserve(numSample);

	std::srand((unsigned int)std::time(NULL));
	for (std::size_t i = 0; i < numSample; ++i)
	{
#if 1
		sampler.sample(x, newX);
		samples.push_back(newX[0]);
		x.swap(newX);
#else
		if (i % 2)
		{
			sampler.sample(newX, x);
			samples.push_back(x[0]);
		}
		else
		{
			sampler.sample(x, newX);
			samples.push_back(newX[0]);
		}
#endif
	}

#if 0
	// output
	for (std::size_t i = 0; i < numSample; ++i)
		std::cout << samples[i] << ' ';
	std::cout << std::endl;

	// in matlab
	// hist(samples, 360)
#endif

#if 1
	// histogram: [min, max] = [-10.0, 20.0], #bins = 150, bin width = 0.2.
	const std::size_t numBin = 150;
	const double binWidth = (20.0 - -10.0) / numBin;
	std::vector<std::size_t> histogram(numBin, 0);
	for (std::size_t i = 0; i < numSample; ++i)
	{
		const int idx = int(samples[i] / binWidth);
		++histogram[idx];
	}

	for (std::size_t i = 0; i < numBin; ++i)
		std::cout << histogram[i] << ' ';
	std::cout << std::endl;

	// in matlab
	// bar(histogram)
#endif
}
