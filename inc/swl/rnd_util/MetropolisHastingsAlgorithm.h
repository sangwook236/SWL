#if !defined(__SWL_RND_UTIL__METROPOLIS_HASTINGS_ALGORITHM__H_)
#define __SWL_RND_UTIL__METROPOLIS_HASTINGS_ALGORITHM__H_ 1


#include "swl/rnd_util/ExportRndUtil.h"
#include <boost/random/linear_congruential.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/numeric/ublas/vector.hpp>


namespace swl {

//--------------------------------------------------------------------------
// Metropolis-Hastings algorithm: Markov chain Monte Carlo method

class SWL_RND_UTIL_API MetropolisHastingsAlgorithm
{
public:
	//typedef MetropolisHastingsAlgorithm base_type;
	typedef boost::numeric::ublas::vector<double> vector_type;

public:
	// target distribution
	struct TargetDistribution
	{
		virtual double evaluate(const vector_type &x, const vector_type *param = NULL) const = 0;
	};

	// proposal distribution
	struct ProposalDistribution
	{
		virtual double evaluate(const vector_type &x, const vector_type &param) const = 0;
		virtual void sample(const vector_type &param, vector_type &sample) const = 0;
	};

public:
	MetropolisHastingsAlgorithm(TargetDistribution &targetDistribution, ProposalDistribution &proposalDistribution);
	~MetropolisHastingsAlgorithm();

private:
	MetropolisHastingsAlgorithm(const MetropolisHastingsAlgorithm &rhs);
	MetropolisHastingsAlgorithm & operator=(const MetropolisHastingsAlgorithm &rhs);

public:
	void sample(const vector_type &x, vector_type &newX) const;

private:
	typedef boost::minstd_rand base_generator_type;
	typedef boost::variate_generator<base_generator_type &, boost::uniform_real<> > generator_type;

private:
	TargetDistribution &targetDistribution_;
	ProposalDistribution &proposalDistribution_;

	base_generator_type baseGenerator_;
	mutable generator_type generator_;
};

}  // namespace swl


#endif  // __SWL_RND_UTIL__METROPOLIS_HASTINGS_ALGORITHM__H_
