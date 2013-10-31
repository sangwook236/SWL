#if !defined(__SWL_RND_UTIL__LOCAL_API__H_)
#define __SWL_RND_UTIL__LOCAL_API__H_ 1


#include "swl/rnd_util/RejectionSampling.h"
#include <boost/random/linear_congruential.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>


namespace swl {

//--------------------------------------------------------------------------
// von Mises target distribution

// [ref] "Pattern Recognition and Machine Learning" by Christopher M. Bishop, ch. 11.1.2

struct VonMisesTargetDistribution: public swl::RejectionSampling::TargetDistribution
{
	typedef swl::RejectionSampling::TargetDistribution base_type;
	typedef base_type::vector_type vector_type;

	VonMisesTargetDistribution()
	: base_type(), mean_direction_(0.0), kappa_(0.0)
	{
	}

	/*virtual*/ double evaluate(const vector_type &x) const;

	void setParameters(const double mean_direction, const double kappa)
	{
		mean_direction_ = mean_direction;
		kappa_ = kappa;
	}

private:
	double mean_direction_;  // the mean directions of the von Mises distribution. 0 <= mu < 2 * pi. [rad].
	double kappa_;  // the concentration parameters of the von Mises distribution. kappa >= 0.
};

//--------------------------------------------------------------------------
// univariate normal proposal distribution

// [ref] "Pattern Recognition and Machine Learning" by Christopher M. Bishop, ch. 11.1.2

struct UnivariateNormalProposalDistribution: public swl::RejectionSampling::ProposalDistribution
{
	typedef swl::RejectionSampling::ProposalDistribution base_type;
	typedef base_type::vector_type vector_type;

	UnivariateNormalProposalDistribution();

	// evaluate k * proposal_distribution(x)
	/*virtual*/ double evaluate(const vector_type &x) const;
	/*virtual*/ void sample(vector_type &sample) const;

	void setParameters(const double mean, const double sigma, const double k = 1.0);
	void setSeed(const unsigned int seed);

private:
	typedef boost::minstd_rand base_generator_type;
	typedef boost::variate_generator<base_generator_type &, boost::normal_distribution<> > generator_type;

private:
	double mean_;  // the mean of the univariate normal distribution
	double sigma_;  // the standard deviation of the univariate normal distribution

	base_generator_type baseGenerator_;
	mutable generator_type generator_;
};

//--------------------------------------------------------------------------
// univariate uniform proposal distribution

// [ref] "Pattern Recognition and Machine Learning" by Christopher M. Bishop, ch. 11.1.2

struct UnivariateUniformProposalDistribution: public swl::RejectionSampling::ProposalDistribution
{
	typedef swl::RejectionSampling::ProposalDistribution base_type;
	typedef base_type::vector_type vector_type;

	UnivariateUniformProposalDistribution();

	// evaluate k * proposal_distribution(x)
	/*virtual*/ double evaluate(const vector_type &x) const;
	/*virtual*/ void sample(vector_type &sample) const;

	void setParameters(const double lower, const double upper, const double k = 1.0);
	void setSeed(const unsigned int seed);

private:
	double lower_, upper_;  // the lower & upper bound of the univariate uniform distribution
};

//--------------------------------------------------------------------------
// find MAP estimate of multinomial using entropic prior.

// [ref] "Structure Learning in Conditional Probability Models via an Entropic Prior and Parameter Extinction", M. Brand, Neural Computation, 1999.
// [ref] "Pattern discovery via entropy minimization", M. Brand, AISTATS, 1999.

bool computeMAPEstimateOfMultinomialUsingEntropicPrior(const std::vector<double> &omega, const double &z, std::vector<double> &theta, double &logLikelihood, const double terminationTolerance, const std::size_t maxIteration, const bool doesInitializeLambdaFirst = true);
bool computeMAPEstimateOfMultinomialUsingEntropicPrior(const boost::numeric::ublas::vector<double> &omega, const double &z, std::vector<double> &theta, double &logLikelihood, const double terminationTolerance, const std::size_t maxIteration, const bool doesInitializeLambdaFirst = true);

}  // namespace swl


#endif  // __SWL_RND_UTIL__LOCAL_API__H_
