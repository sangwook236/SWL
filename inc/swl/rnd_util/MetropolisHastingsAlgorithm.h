#if !defined(__SWL_RND_UTIL__METROPOLIS_HASTINGS_ALGORITHM__H_)
#define __SWL_RND_UTIL__METROPOLIS_HASTINGS_ALGORITHM__H_ 1


#include "swl/rnd_util/ExportRndUtil.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_blas.h>


namespace swl {

//--------------------------------------------------------------------------
// Metropolis-Hastings algorithm: Markov chain Monte Carlo method

class SWL_RND_UTIL_API MetropolisHastingsAlgorithm
{
public:
	//typedef MetropolisHastingsAlgorithm base_type;

	// target distribution
	struct TargetDistribution
	{
		virtual double evaluate(const gsl_vector *x, const gsl_vector *param = NULL) const = 0;
	};

	// proposal distribution
	struct ProposalDistribution
	{
		virtual double evaluate(const gsl_vector *x, const gsl_vector *param) const = 0;
		virtual void sample(const gsl_vector *param, gsl_vector *sample) const = 0;
	};

public:
	MetropolisHastingsAlgorithm(const gsl_vector *x0, TargetDistribution &targetDistribution, ProposalDistribution &proposalDistribution);
	~MetropolisHastingsAlgorithm();

private:
	MetropolisHastingsAlgorithm(const MetropolisHastingsAlgorithm &rhs);
	MetropolisHastingsAlgorithm & operator=(const MetropolisHastingsAlgorithm &rhs);

public:
	const gsl_vector * sample();

private:
	const size_t dim_;

	TargetDistribution &targetDistribution_;
	ProposalDistribution &proposalDistribution_;

	gsl_vector *x_;
	gsl_vector *x_star_;

	gsl_rng *r_;
};

}  // namespace swl


#endif  // __SWL_RND_UTIL__METROPOLIS_HASTINGS_ALGORITHM__H_
