#include "stdafx.h"
#include "swl/Config.h"
#include "swl/rnd_util/MetropolisHastingsAlgorithm.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <vector>
#include <cmath>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace {

struct TargetDistribution: public swl::MetropolisHastingsAlgorithm::TargetDistribution
{
	/*virtual*/ double evaluate(const gsl_vector *x, const gsl_vector *param = NULL) const
	{
		const double v = gsl_vector_get(x, 0);
		return 0.3 * std::exp(-0.2 * v * v) + 0.7 * std::exp(-0.2 * (v - 10.0) * (v - 10.0));
	}
};

struct ProposalDistribution: public swl::MetropolisHastingsAlgorithm::ProposalDistribution
{
	/*virtual*/ double evaluate(const gsl_vector *x, const gsl_vector *param) const
	{
		const double v = gsl_vector_get(x, 0);
		const double mean = gsl_vector_get(param, 0);
		return gsl_ran_gaussian_pdf(v - mean, sigma_);
	}
	/*virtual*/ void sample(const gsl_vector *param, gsl_vector *sample) const
	{
		gsl_rng_env_setup();
		const gsl_rng_type *T = gsl_rng_default;
		gsl_rng *r = gsl_rng_alloc(T);

		const double mean = gsl_vector_get(param, 0);
		const double prob = mean + gsl_ran_gaussian(r, sigma_);

		gsl_rng_free(r);

		gsl_vector_set(sample, 0, prob);
	}

private:
	static const double sigma_;
};

/*static*/ const double ProposalDistribution::sigma_ = 100.0;

const double HISTO_MIN = -10.0;
const double HISTO_MAX = 20.0;
const size_t HISTO_COUNT = 150;

void constructHistogram(const double v, std::vector<int> &histo)
{
	if (v < HISTO_MIN || v >= HISTO_MAX) return;

	const double h = (HISTO_MAX - HISTO_MIN) / HISTO_COUNT;
	const size_t idx = size_t((v - HISTO_MIN) / h);
	++histo[idx];
}

}  // unnamed namespace

void metropolis_hastings_algorithm()
{
	const size_t STATE_DIM = 1;

	gsl_vector *x0 = gsl_vector_alloc(STATE_DIM);
	gsl_vector_set_zero(x0);

	//
	gsl_rng_env_setup();

	swl::MetropolisHastingsAlgorithm mh(x0, TargetDistribution(), ProposalDistribution());

	gsl_vector_free(x0);

	const size_t Niteration = 100;
	std::vector<int> histogram(HISTO_COUNT, 0);

	for (size_t i = 0; i < Niteration; ++i)
	{
		const gsl_vector *x = mh.sample();
		const double v = gsl_vector_get(x, 0);
		constructHistogram(v, histogram);
	}
}
