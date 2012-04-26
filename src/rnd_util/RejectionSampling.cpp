#include "swl/Config.h"
#include "swl/rnd_util/RejectionSampling.h"
#include <boost/random/linear_congruential.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <ctime>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//--------------------------------------------------------------------------
// rejection sampling

// [ref] "Pattern Recognition and Machine Learning" by Christopher M. Bishop, ch. 11.1.2

RejectionSampling::RejectionSampling(TargetDistribution &targetDistribution, ProposalDistribution &proposalDistribution)
: targetDistribution_(targetDistribution), proposalDistribution_(proposalDistribution)
{
}

RejectionSampling::~RejectionSampling()
{
}

bool RejectionSampling::sample(vector_type &x, const std::size_t maxIteration /*= 100*/) const
{
	// PRECONDITIONS [] >>
	//	-. std::srand() had to be called before this function is called.

	std::size_t iter = 1;
	do
	{
		proposalDistribution_.sample(x);
		const double k_q(proposalDistribution_.evaluate(x));
		const double p(targetDistribution_.evaluate(x));

		const double u0(((double)std::rand() / RAND_MAX) * k_q);

		if (u0 <= p) return true;

		++iter;
	} while (iter <= maxIteration);

	return false;
}

}  // namespace swl
