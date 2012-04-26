#if !defined(__SWL_RND_UTIL__REJECTION_SAMPLING__H_)
#define __SWL_RND_UTIL__REJECTION_SAMPLING__H_ 1


#include "swl/rnd_util/ExportRndUtil.h"
#include <boost/numeric/ublas/vector.hpp>


namespace swl {

//--------------------------------------------------------------------------
// rejection sampling

class SWL_RND_UTIL_API RejectionSampling
{
public:
	//typedef RejectionSampling base_type;
	typedef boost::numeric::ublas::vector<double> vector_type;

public:
	// target distribution
	struct TargetDistribution
	{
		typedef RejectionSampling::vector_type vector_type;

		virtual double evaluate(const vector_type &x) const = 0;
	};

	// proposal distribution
	struct ProposalDistribution
	{
		typedef RejectionSampling::vector_type vector_type;

		ProposalDistribution(const double k = 1.0)
		: k_(k)
		{}

		// evaluate k * proposal_distribution(x)
		virtual double evaluate(const vector_type &x) const = 0;
		virtual void sample(vector_type &sample) const = 0;

	protected:
		const double k_;  // a constant. k * proposal_distribution(x) >= target_distribution(x).
	};

public:
	RejectionSampling(TargetDistribution &targetDistribution, ProposalDistribution &proposalDistribution);
	~RejectionSampling();

private:
	RejectionSampling(const RejectionSampling &rhs);  // not implemented
	RejectionSampling & operator=(const RejectionSampling &rhs);  // not implemented

public:
	bool sample(vector_type &x, const std::size_t maxIteration = 100) const;

private:
	TargetDistribution &targetDistribution_;
	ProposalDistribution &proposalDistribution_;
};

}  // namespace swl


#endif  // __SWL_RND_UTIL__REJECTION_SAMPLING__H_
