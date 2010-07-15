#include "swl/Config.h"
#include "swl/rnd_util/MetropolisHastingsAlgorithm.h"
#include <boost/random/linear_congruential.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <stdexcept>
#include <ctime>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//--------------------------------------------------------------------------
// Metropolis-Hastings algorithm: Markov chain Monte Carlo method

// [ref]
// "An Introduction to MCMC for Machine Learning", Christophe Andrieu, Nando de Freitas, Arnaud Doucet, and Michael I. Jordan
//	Machine Learning, 50, pp. 5-43, 2003

MetropolisHastingsAlgorithm::MetropolisHastingsAlgorithm(TargetDistribution &targetDistribution, ProposalDistribution &proposalDistribution)
: targetDistribution_(targetDistribution), proposalDistribution_(proposalDistribution),
  baseGenerator_(static_cast<unsigned int>(std::time(NULL))), generator_(baseGenerator_, boost::uniform_real<>(0, 1))
{
	//baseGenerator_.seed(static_cast<unsigned int>(std::time(NULL)));
}

MetropolisHastingsAlgorithm::~MetropolisHastingsAlgorithm()
{
}

void MetropolisHastingsAlgorithm::sample(const vector_type &x, vector_type &newX) const
{
	proposalDistribution_.sample(x, newX);

	const double p_i = targetDistribution_.evaluate(x);
	const double p_star = targetDistribution_.evaluate(newX);
	const double q_i = proposalDistribution_.evaluate(x, newX);
	const double q_star = proposalDistribution_.evaluate(newX, x);

	const double eps = 1.0e-15;

	const double num = p_star * q_i;
	const double den = p_i * q_star;

	// TODO [check] >>
	const double A = den > eps ? std::min(1.0, num / den) : 1.0;

	const double &u = generator_();

	if (u >= A) newX = x;
}

}  // namespace swl
