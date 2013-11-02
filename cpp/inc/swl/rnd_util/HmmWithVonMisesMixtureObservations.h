#if !defined(__SWL_RND_UTIL__HMM_WITH_VON_MISES_MIXTURE_OBSERVATIONS__H_)
#define __SWL_RND_UTIL__HMM_WITH_VON_MISES_MIXTURE_OBSERVATIONS__H_ 1


#include "swl/rnd_util/CDHMMWithMixtureObservations.h"
#include <boost/smart_ptr.hpp>


namespace swl {

struct VonMisesTargetDistribution;
struct UnivariateNormalProposalDistribution;
struct UnivariateUniformProposalDistribution;

//--------------------------------------------------------------------------
// continuous density HMM with von Mises mixture observation densities.

class SWL_RND_UTIL_API HmmWithVonMisesMixtureObservations: public CDHMMWithMixtureObservations
{
public:
	typedef CDHMMWithMixtureObservations base_type;

public:
	HmmWithVonMisesMixtureObservations(const size_t K, const size_t C);  // for ML learning.
	HmmWithVonMisesMixtureObservations(const size_t K, const size_t C, const dvector_type &pi, const dmatrix_type &A, const dmatrix_type &alphas, const dmatrix_type &mus, const dmatrix_type &kappas);
	HmmWithVonMisesMixtureObservations(const size_t K, const size_t C, const dvector_type *pi_conj, const dmatrix_type *A_conj, const dmatrix_type *alphas_conj, const dmatrix_type *ms_conj, const dmatrix_type *Rs_conj, const dmatrix_type *cs_conj);  // for MAP learning using conjugate prior.
	virtual ~HmmWithVonMisesMixtureObservations();

private:
	HmmWithVonMisesMixtureObservations(const HmmWithVonMisesMixtureObservations &rhs);  // not implemented.
	HmmWithVonMisesMixtureObservations & operator=(const HmmWithVonMisesMixtureObservations &rhs);  // not implemented.

public:
	//
	dmatrix_type & getMeanDirection()  {  return mus_;  }
	const dmatrix_type & getMeanDirection() const  {  return mus_;  }
	dmatrix_type & getConcentrationParameter()  {  return  kappas_;  }
	const dmatrix_type & getConcentrationParameter() const  {  return  kappas_;  }

protected:
	// if state == 0, hidden state = [ 1 0 0 ... 0 0 ].
	// if state == 1, hidden state = [ 0 1 0 ... 0 0 ].
	// ...
	// if state == N-1, hidden state = [ 0 0 0 ... 0 1 ].
	/*virtual*/ double doEvaluateEmissionMixtureComponentProbability(const unsigned int state, const unsigned int component, const dvector_type &observation) const;

	//
	/*virtual*/ void doGenerateObservationsSymbol(const unsigned int state, const size_t n, dmatrix_type &observations) const;
	// if seed != -1, the seed value is set.
	/*virtual*/ void doInitializeRandomSampleGeneration(const unsigned int seed = (unsigned int)-1) const;
	/*virtual*/ void doFinalizeRandomSampleGeneration() const;

	// ML learning.
	//	-. for a single independent observation sequence.
	/*virtual*/ void doEstimateObservationDensityParametersByML(const size_t N, const unsigned int state, const dmatrix_type &observations, const dmatrix_type &gamma, const double denominatorA);
	//	-. for multiple independent observation sequences.
	/*virtual*/ void doEstimateObservationDensityParametersByML(const std::vector<size_t> &Ns, const unsigned int state, const std::vector<dmatrix_type> &observationSequences, const std::vector<dmatrix_type> &gammas, const size_t R, const double denominatorA);

	// MAP learning using conjugate prior.
	//	-. for a single independent observation sequence.
	/*virtual*/ void doEstimateObservationDensityParametersByMAPUsingConjugatePrior(const size_t N, const unsigned int state, const dmatrix_type &observations, const dmatrix_type &gamma, const double denominatorA);
	//	-. for multiple independent observation sequences.
	/*virtual*/ void doEstimateObservationDensityParametersByMAPUsingConjugatePrior(const std::vector<size_t> &Ns, const unsigned int state, const std::vector<dmatrix_type> &observationSequences, const std::vector<dmatrix_type> &gammas, const size_t R, const double denominatorA);

	// MAP learning using entropic prior.
	//	-. for a single independent observation sequence.
	/*virtual*/ void doEstimateObservationDensityParametersByMAPUsingEntropicPrior(const size_t N, const unsigned int state, const dmatrix_type &observations, const dmatrix_type &gamma, const double z, const bool doesTrimParameter, const double terminationTolerance, const size_t maxIteration, const double /*denominatorA*/);
	//	-. for multiple independent observation sequences.
	/*virtual*/ void doEstimateObservationDensityParametersByMAPUsingEntropicPrior(const std::vector<size_t> &Ns, const unsigned int state, const std::vector<dmatrix_type> &observationSequences, const std::vector<dmatrix_type> &gammas, const double z, const bool doesTrimParameter, const double terminationTolerance, const size_t maxIteration, const size_t R, const double /*denominatorA*/);

	//
	/*virtual*/ bool doReadObservationDensity(std::istream &stream);
	/*virtual*/ bool doWriteObservationDensity(std::ostream &stream) const;
	/*virtual*/ void doInitializeObservationDensity(const std::vector<double> &lowerBoundsOfObservationDensity, const std::vector<double> &upperBoundsOfObservationDensity);
	/*virtual*/ void doNormalizeObservationDensityParameters()
	{
		CDHMMWithMixtureObservations::normalizeObservationDensityParameters(K_);
	}

	/*virtual*/ bool doDoHyperparametersOfConjugatePriorExist() const
	{
		return base_type::doDoHyperparametersOfConjugatePriorExist() &&
			NULL != ms_conj_.get() && NULL != Rs_conj_.get() && NULL != cs_conj_.get();
	}

private:
	dmatrix_type mus_;  // the sets of mean directions of each components in the von Mises mixture distribution.
	dmatrix_type kappas_;  // the sets of concentration parameters of each components in the von Mises mixture distribution.

	// hyperparameters for the conjugate prior.
	//	[ref] "EM Algorithm 3 - THE EM Algorithm for MAP Estimates of HMM", personal note.
	//	[ref] "Finding the Location of a Signal: A Bayesian Analysis", Peter Guttorp and Richard A. Lockhart, JASA, 1988.
	boost::scoped_ptr<const dmatrix_type> ms_conj_;  // m.
	boost::scoped_ptr<const dmatrix_type> Rs_conj_;  // R. R >= 0.
	boost::scoped_ptr<const dmatrix_type> cs_conj_;  // c. non-negative integer.

	mutable boost::scoped_ptr<VonMisesTargetDistribution> targetDist_;
#if 0
	mutable boost::scoped_ptr<UnivariateNormalProposalDistribution> proposalDist_;
#else
	mutable boost::scoped_ptr<UnivariateUniformProposalDistribution> proposalDist_;
#endif
};

}  // namespace swl


#endif  // __SWL_RND_UTIL__HMM_WITH_VON_MISES_MIXTURE_OBSERVATIONS__H_
