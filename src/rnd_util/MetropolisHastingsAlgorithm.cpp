#include "swl/Config.h"
#include "swl/rnd_util/MetropolisHastingsAlgorithm.h"
#include <gsl/gsl_randist.h>
#include <stdexcept>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

MetropolisHastingsAlgorithm::MetropolisHastingsAlgorithm(const gsl_vector *x0, TargetDistribution &targetDistribution, ProposalDistribution &proposalDistribution)
: dim_(x0 ? x0->size : 0), targetDistribution_(targetDistribution), proposalDistribution_(proposalDistribution),
  x_(NULL), x_star_(NULL), r_(NULL)
{
	if (NULL == x0 || 0 == dim_)
		throw std::runtime_error("contruction error");

	x_ = gsl_vector_alloc(dim_);
	gsl_vector_memcpy(x_, x0);
	x_star_ = gsl_vector_alloc(dim_);
	gsl_vector_set_zero(x_star_);

	const gsl_rng_type *T = gsl_rng_default;
	r_ = gsl_rng_alloc(T);
}

MetropolisHastingsAlgorithm::~MetropolisHastingsAlgorithm()
{
	gsl_vector_free(x_);  x_ = NULL;
	gsl_vector_free(x_star_);  x_star_ = NULL;

	gsl_rng_free(r_);
}

const gsl_vector * MetropolisHastingsAlgorithm::sample()
{
	proposalDistribution_.sample(x_, x_star_);

	const double p_i = targetDistribution_.evaluate(x_);
	const double p_star = targetDistribution_.evaluate(x_star_);
	const double q_i = proposalDistribution_.evaluate(x_, x_star_);
	const double q_star = proposalDistribution_.evaluate(x_star_, x_);

	const double eps = 1.0e-15;

	const double num = p_star * q_i;
	const double den = p_i * q_star;
	const double A = den > eps ? std::min(1.0, num / den) : 1.0;

	const double u = gsl_ran_flat(r_, 0.0, 1.0);

	if (u < A) gsl_vector_memcpy(x_, x_star_);

	return x_;
}

}  // namespace swl
