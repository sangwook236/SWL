//#include "stdafx.h"
#include "swl/Config.h"
#include "swl/rnd_util/SamplingImportanceResampling.h"
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
// "On sequential Monte Carlo sampling methods for Bayesian filtering", Arnaud Doucet, Simon Godsill, and Christophe Andrieu,
//	Statistics and Computing, 10, pp. 197-208, 2000  ==>  pp. 206
// "Novel approach to nonlinear/non-Gaussian Bayesian state estimation", N. J. Gordon, D. J. Salmond, and A. F. M. Smith,
//	IEE Proceedings, vol. 140, no. 2, pp. 107-113, 1993  ==>  pp. 109
// "A Tutorial on Particle Filters for Online Nonlinear/Non-Gaussian Bayesian Tracking", M. S. Arulampalam, Simon Maskell, Neil Gordon, and Time Clapp,
//	Trans. on Signal Processing, vol. 50, no. 2, pp. 174-188, 2002  ==>  pp. 183
// "An Introduction to Sequential Monte Carlo Methods", Arnaud Doucet, Nando de Freitas, and Neil Gordon, 2001  ==>  pp. 12

struct TransitionDistribution: public swl::SamplingImportanceResampling::TransitionDistribution
{
	TransitionDistribution(const double Ts, const double sigma)
	: Ts_(Ts), sigma_(sigma)
	{}

	// p(x(k) | x(k-1))
	/*virtual*/ double evaluate(const size_t step, const swl::SamplingImportanceResampling::vector_type &currX, const swl::SamplingImportanceResampling::vector_type &prevX) const
	{
		const double &x0 = prevX[0];
		const double &x1 = currX[0];
		const double f = 0.5 * x0 + 25.0 * x0 / (1 + x0*x0) + 8.0 * std::cos(1.2 * Ts_ * step);

		boost::math::normal dist(f, sigma_);
		return boost::math::pdf(dist, x1);
	}

private:
	const double Ts_;
	const double sigma_;
};

struct ObservationDistribution: public swl::SamplingImportanceResampling::ObservationDistribution
{
	ObservationDistribution(const double Ts, const double sigma)
	: Ts_(Ts), sigma_(sigma)
	{}

	// p(y(k) | x(k))
	/*virtual*/ double evaluate(const size_t step, const swl::SamplingImportanceResampling::vector_type &x, const swl::SamplingImportanceResampling::vector_type &y) const
	{
		const double &x0 = x[0];
		const double &y0 = y[0];
		const double g = x0 * x0 / 20.0;

		boost::math::normal dist(g, sigma_);
		return boost::math::pdf(dist, y0);
	}

private:
	const double Ts_;
	const double sigma_;
};

struct ProposalDistribution: public swl::SamplingImportanceResampling::ProposalDistribution
{
	ProposalDistribution(const double Ts, const double sigma_v, const double sigma_w)
	: Ts_(Ts), sigma_v_(sigma_v), sigma_w_(sigma_w),
	  baseGenerator_(static_cast<unsigned int>(std::time(NULL)))
	{}

	// p(x(k) | x(k-1))
	/*virtual*/ double evaluate(const size_t step, const swl::SamplingImportanceResampling::vector_type &currX, const swl::SamplingImportanceResampling::vector_type &prevX, const swl::SamplingImportanceResampling::vector_type &y) const
	{
		const double &x0 = prevX[0];
		const double &x1 = currX[0];
		const double &y0 = y[0];

		const double f = 0.5 * x0 + 25.0 * x0 / (1 + x0*x0) + 8.0 * std::cos(1.2 * Ts_ * step);

#if 0
		// the prior distribution of a HMM is used as importance function  -->  sub-optimal
		//	use transition distribution
		boost::math::normal dist(f, sigma_);
		return boost::math::pdf(dist, x1);
#else
		// an importance function obtained by local linearization
		const double sigma = 1.0 / std::sqrt(1.0 / (sigma_v_ * sigma_v_) + f * f / (100.0 * sigma_w_ * sigma_w_));
		const double mean = sigma*sigma * (f / (sigma_v_ * sigma_v_) + (f / (10.0 * sigma_w_ * sigma_w_)) * (y0 + f * f / 20.0));

		boost::math::normal dist(mean, sigma);
		return boost::math::pdf(dist, x1);
#endif
	}
	// p(x(k) | x(0:k-1), y(0:k))
	/*virtual*/ double evaluate(const size_t step, const swl::SamplingImportanceResampling::vector_type &currX, const std::vector<swl::SamplingImportanceResampling::vector_type> &prevXs, const std::vector<swl::SamplingImportanceResampling::vector_type> &simulatedYs) const
	{  throw std::runtime_error("this function doesn't have to be called");  }

	// x ~ p(x(k) | x(k-1))
	/*virtual*/ void sample(const size_t step, const swl::SamplingImportanceResampling::vector_type &x, const swl::SamplingImportanceResampling::vector_type &y, swl::SamplingImportanceResampling::vector_type &sample) const
	{
		const double &x0 = x[0];
		const double &y0 = y[0];

		const double f = 0.5 * x0 + 25.0 * x0 / (1 + x0*x0) + 8.0 * std::cos(1.2 * Ts_ * step);

#if 0
		// the prior distribution of a HMM is used as importance function  -->  sub-optimal
		//	use transition distribution
		generator_type generator(baseGenerator_, boost::normal_distribution<>(f, sigma_v_));
		sample[0] = generator();
#else
		// an importance function obtained by local linearization
		const double sigma = 1.0 / std::sqrt(1.0 / (sigma_v_ * sigma_v_) + f * f / (100.0 * sigma_w_ * sigma_w_));
		const double mean = sigma*sigma * (f / (sigma_v_ * sigma_v_) + (f / (10.0 * sigma_w_ * sigma_w_)) * (y0 + f * f / 20.0));

		generator_type generator(baseGenerator_, boost::normal_distribution<>(mean, sigma));
		sample[0] = generator();
#endif
	}
	// x ~ p(x(k) | x(0:k-1), y(0:k))
	/*virtual*/ void sample(const size_t step, const std::vector<swl::SamplingImportanceResampling::vector_type> &xs, const std::vector<swl::SamplingImportanceResampling::vector_type> &simulatedYs, swl::SamplingImportanceResampling::vector_type &sample) const
	{  throw std::runtime_error("this function doesn't have to be called");  }

private:
	typedef boost::minstd_rand base_generator_type;
	typedef boost::variate_generator<base_generator_type &, boost::normal_distribution<> > generator_type;

private:
	const double Ts_;
	const double sigma_v_;
	const double sigma_w_;

	mutable base_generator_type baseGenerator_;
};

}  // namespace local
}  // unnamed namespace

void sampling_importance_resampling()
{
	const size_t PARTICLE_NUM = 1000;
	const double EFFECTIVE_SAMPLE_SIZE = 1000.0;
	const size_t STATE_DIM = 1;
	const size_t OUTPUT_DIM = 1;

	const double Ts = 1.0;
	const double x_init = 0.1;
	const double sigma_init = std::sqrt(2.0);
	const double sigma_v = std::sqrt(10.0);
	const double sigma_w = 1.0;

	//
	const size_t Nstep = 50;

	// simulate measurement
	std::vector<double> simulatedXs, simulatedYs;
	simulatedXs.reserve(Nstep);
	simulatedYs.reserve(Nstep);
	{
		// x ~ p(x(0))
		typedef boost::minstd_rand base_generator_type;
		typedef boost::variate_generator<base_generator_type &, boost::normal_distribution<> > generator_type;

		base_generator_type baseGenerator(static_cast<unsigned int>(std::time(NULL)));
		generator_type generator1(baseGenerator, boost::normal_distribution<>(0.0, sigma_v));
		generator_type generator2(baseGenerator, boost::normal_distribution<>(0.0, sigma_w));

		const double x0 = 0.1;
		double x = x0;
		for (size_t step = 1; step <= Nstep; ++step)
		{
			const double f = 0.5 * x + 25.0 * x / (1 + x*x) + 8.0 * std::cos(1.2 * Ts * step);
			x = f + generator1();
			simulatedXs.push_back(x);
			//simulatedYs.push_back(x * x / 20.0);
			simulatedYs.push_back(x * x / 20.0 + generator2());
		}
	}

	//
#if defined(__GNUC__)
    local::TransitionDistribution transitionDistribution(Ts, sigma_v);
	local::ObservationDistribution observationDistribution(Ts, sigma_w);
	local::ProposalDistribution proposalDistribution(Ts, sigma_v, sigma_w);
	swl::SamplingImportanceResampling sir(EFFECTIVE_SAMPLE_SIZE, transitionDistribution, observationDistribution, proposalDistribution);
#else
	swl::SamplingImportanceResampling sir(EFFECTIVE_SAMPLE_SIZE, local::TransitionDistribution(Ts, sigma_v), local::ObservationDistribution(Ts, sigma_w), local::ProposalDistribution(Ts, sigma_v, sigma_w));
#endif

	std::vector<swl::SamplingImportanceResampling::vector_type> xs(PARTICLE_NUM, swl::SamplingImportanceResampling::vector_type(STATE_DIM, x_init));
	std::vector<swl::SamplingImportanceResampling::vector_type> newXs(PARTICLE_NUM, swl::SamplingImportanceResampling::vector_type(STATE_DIM, x_init));
	std::vector<double> weights(PARTICLE_NUM, 1.0 / PARTICLE_NUM);
	swl::SamplingImportanceResampling::vector_type y(OUTPUT_DIM, 0.0);

	// initialization: step = 0
	{
		// x ~ p(x(0))
		typedef boost::minstd_rand base_generator_type;
		typedef boost::variate_generator<base_generator_type &, boost::normal_distribution<> > generator_type;

		base_generator_type baseGenerator(static_cast<unsigned int>(std::time(NULL)));
		generator_type generator(baseGenerator, boost::normal_distribution<>(0.0, sigma_init));

		for (size_t i = 0; i < PARTICLE_NUM; ++i)
			xs[i][0] = generator();
	}

	//
	swl::SamplingImportanceResampling::vector_type estimatedX(STATE_DIM);
	std::vector<double> estimatedXs;
	estimatedXs.reserve(Nstep);
	std::vector<double> histogram(PARTICLE_NUM, 0.0);

	for (size_t step = 1; step <= Nstep; ++step)
	{
		y[0] = simulatedYs[step - 1];

#if 0
		sir.sample(step, PARTICLE_NUM, xs, y, newXs, weights, &estimatedX);

		// estimated state  ==>  a particle with a maximum weight
		estimatedXs.push_back(estimatedX[0]);
#elif 0
		sir.sample(step, PARTICLE_NUM, xs, y, newXs, weights);

		// estimated state  ==>  the mean of the posterior samples (particles)

		// arithmetic mean
		double sum = 0.0;
		for (size_t k = 0; k < PARTICLE_NUM; ++k)
			sum += newXs[k][0];
		estimatedXs.push_back(sum / PARTICLE_NUM);
#elif 1
		sir.sample(step, PARTICLE_NUM, xs, y, newXs, weights);

		// estimated state  ==>  the mean of the posterior samples (particles)

		// weighted mean
		double mean = 0.0;
		for (size_t k = 0; k < PARTICLE_NUM; ++k)
			mean += weights[k] * newXs[k][0];
		estimatedXs.push_back(mean);
#endif

		for (size_t k = 0; k < PARTICLE_NUM; ++k)
			histogram[k] = newXs[k][0];

		newXs.swap(xs);
	}
}
