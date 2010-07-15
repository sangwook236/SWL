#include "stdafx.h"
#include "swl/Config.h"
#include "swl/rnd_util/MetropolisHastingsAlgorithm.h"
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

// [ref]
// "An Introduction to MCMC for Machine Learning", Christophe Andrieu, Nando de Freitas, Arnaud Doucet, and Michael I. Jordan
//	Machine Learning, 50, pp. 5-43, 2003

struct TargetDistribution: public swl::MetropolisHastingsAlgorithm::TargetDistribution
{
	/*virtual*/ double evaluate(const swl::MetropolisHastingsAlgorithm::vector_type &x, const swl::MetropolisHastingsAlgorithm::vector_type *param = NULL) const
	{
		const double &v = x[0];
		return 0.3 * std::exp(-0.2 * v * v) + 0.7 * std::exp(-0.2 * (v - 10.0) * (v - 10.0));
	}
};

struct ProposalDistribution: public swl::MetropolisHastingsAlgorithm::ProposalDistribution
{
	ProposalDistribution(const double sigma)
	: sigma_(sigma), baseGenerator_(static_cast<unsigned int>(std::time(NULL))), generator_(baseGenerator_, boost::normal_distribution<>(0.0, sigma_))
	{}

	/*virtual*/ double evaluate(const swl::MetropolisHastingsAlgorithm::vector_type &x, const swl::MetropolisHastingsAlgorithm::vector_type &param) const
	{
		const double &mean = param[0];
		boost::math::normal dist(mean, sigma_);
		return boost::math::pdf(dist, x[0]);
	}
	/*virtual*/ void sample(const swl::MetropolisHastingsAlgorithm::vector_type &param, swl::MetropolisHastingsAlgorithm::vector_type &sample) const
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

}  // unnamed namespace

void metropolis_hastings_algorithm()
{
	const size_t STATE_DIM = 1;
	const double sigma = 10.0;

	TargetDistribution targetDist;
	ProposalDistribution proposalDist(sigma);
	swl::MetropolisHastingsAlgorithm mha(targetDist, proposalDist);

	swl::MetropolisHastingsAlgorithm::vector_type x(STATE_DIM, 0.0);
	swl::MetropolisHastingsAlgorithm::vector_type newX(STATE_DIM, 0.0);

	//
	const size_t Nstep = 10000;

	// [min, max] = [-10.0, 20.0], #bins = 150, bin width = 0.2
	std::vector<double> histogram;
	histogram.reserve(Nstep);

	for (size_t i = 0; i < Nstep; ++i)
	{
#if 1
		mha.sample(x, newX);
		histogram.push_back(newX[0]);
		x.swap(newX);
#else
		if (i % 2)
		{
			mha.sample(newX, x);
			histogram.push_back(x[0]);
		}
		else
		{
			mha.sample(x, newX);
			histogram.push_back(newX[0]);
		}
#endif
	}
}
