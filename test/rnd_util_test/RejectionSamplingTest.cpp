//#include "stdafx.h"
#include "swl/Config.h"
#include "swl/math/MathUtil.h"
#include "swl/rnd_util/RejectionSampling.h"
#include <boost/random/linear_congruential.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/constants/constants.hpp>
#include <iostream>
#include <vector>
#include <cmath>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace {
namespace local {

// [ref] "Pattern Recognition and Machine Learning" by Christopher M. Bishop, ch. 11.1.2

struct VonMisesDistribution: public swl::RejectionSampling::TargetDistribution
{
	typedef swl::RejectionSampling::TargetDistribution base_type;
	typedef base_type::vector_type vector_type;

	VonMisesDistribution(const double mean_direction, const double kappa)
	: base_type(), mean_direction_(mean_direction), kappa_(kappa)
	{
	}

	/*virtual*/ double evaluate(const vector_type &x) const
	{
		return 0.5 * std::exp(kappa_ * std::cos(x[0] - mean_direction_)) / (boost::math::constants::pi<double>() * boost::math::cyl_bessel_i(0.0, kappa_));;
	}

private:
	const double mean_direction_;  // the mean directions of the von Mises distribution. 0 <= mu < 2 * pi. [rad].
	const double kappa_;  // the concentration parameters of the von Mises distribution. kappa >= 0.
};

struct UnivariateNormalProposalDistribution: public swl::RejectionSampling::ProposalDistribution
{
	typedef swl::RejectionSampling::ProposalDistribution base_type;
	typedef base_type::vector_type vector_type;

	UnivariateNormalProposalDistribution(const double mean, const double sigma, const double k = 1.0)
	: base_type(k), mean_(mean), sigma_(sigma), baseGenerator_(static_cast<unsigned int>(std::time(NULL))), generator_(baseGenerator_, boost::normal_distribution<>(mean_, sigma_))
	{}

	// evaluate k * proposal_distribution(x)
	/*virtual*/ double evaluate(const vector_type &x) const
	{
		boost::math::normal dist(mean_, sigma_);
		return k_ * boost::math::pdf(dist, x[0]);
	}
	/*virtual*/ void sample(vector_type &sample) const
	{
		// 0 <= x < 2 * pi
		sample[0] = swl::MathUtil::wrap(generator_(), 0.0, 2.0 * boost::math::constants::pi<double>());
	}

private:
	typedef boost::minstd_rand base_generator_type;
	typedef boost::variate_generator<base_generator_type &, boost::normal_distribution<> > generator_type;

private:
	const double mean_;  // the mean of the univariate normal distribution
	const double sigma_;  // the standard deviation of the univariate normal distribution

	base_generator_type baseGenerator_;
	mutable generator_type generator_;
};

}  // namespace local
}  // unnamed namespace

void rejection_sampling()
{
	const std::size_t STATE_DIM = 1;
	const double mean_direction = 0.0;
	const double kappa = 1.0;
	const double mean = mean_direction;
	const double sigma = 1.55;
	const double k = 1.472;

	local::VonMisesDistribution targetDist(mean_direction, kappa);
	local::UnivariateNormalProposalDistribution proposalDist(mean, sigma, k);
	swl::RejectionSampling sampler(targetDist, proposalDist);

	//
	const std::size_t numSample = 1000;

	std::vector<double> samples;
	samples.reserve(numSample);
	swl::RejectionSampling::vector_type x(STATE_DIM, 0.0);
	const std::size_t maxIteration = 1000;

	std::srand((unsigned int)std::time(NULL));
	for (std::size_t i = 0; i < numSample; ++i)
	{
		const bool retval = sampler.sample(x, maxIteration);
		assert(retval);
		samples.push_back(x[0]);
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
	// histogram: [min, max) = [0.0, 2.0 * pi), #bins = 360, bin width = ?.
	const std::size_t numBin = 360;
	const double binWidth = 2.0 * boost::math::constants::pi<double>() / numBin;
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
