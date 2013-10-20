#if !defined(__SWL_RND_UTIL__HMM_WITH_MULTINOMIAL_OBSERVATIONS__H_)
#define __SWL_RND_UTIL__HMM_WITH_MULTINOMIAL_OBSERVATIONS__H_ 1


#include "swl/rnd_util/DDHMM.h"


namespace swl {

//--------------------------------------------------------------------------
// HMM with discrete multinomial observation densities.

class SWL_RND_UTIL_API HmmWithMultinomialObservations: public DDHMM
{
public:
	typedef DDHMM base_type;

public:
	HmmWithMultinomialObservations(const size_t K, const size_t D);  // for ML learning.
	HmmWithMultinomialObservations(const size_t K, const size_t D, const dvector_type &pi, const dmatrix_type &A, const dmatrix_type &B);
	HmmWithMultinomialObservations(const size_t K, const size_t D, const dvector_type *pi_conj, const dmatrix_type *A_conj, const dmatrix_type *B_conj);  // for MAP learning using conjugate prior.
	virtual ~HmmWithMultinomialObservations();

private:
	HmmWithMultinomialObservations(const HmmWithMultinomialObservations &rhs);  // not implemented.
	HmmWithMultinomialObservations & operator=(const HmmWithMultinomialObservations &rhs);  // not implemented.

public:
	//
	dmatrix_type & getObservationProbabilityMatrix()  {  return B_;  }
	const dmatrix_type & getObservationProbabilityMatrix() const  {  return B_;  }

protected:
	// if state == 0, hidden state = [ 1 0 0 ... 0 0 ].
	// if state == 1, hidden state = [ 0 1 0 ... 0 0 ].
	// ...
	// if state == N-1, hidden state = [ 0 0 0 ... 0 1 ].
	/*virtual*/ double doEvaluateEmissionProbability(const unsigned int state, const unsigned int observation) const
	{  return B_(state, observation);  }

	/*virtual*/ unsigned int doGenerateObservationsSymbol(const unsigned int state) const;
	/*virtual*/ void doInitializeRandomSampleGeneration(const unsigned int seed = (unsigned int)-1) const;
	///*virtual*/ void doFinalizeRandomSampleGeneration() const;

	// ML learning.
	//	-. for a single independent observation sequence.
	/*virtual*/ void doEstimateObservationDensityParametersByML(const size_t N, const unsigned int state, const uivector_type &observations, const dmatrix_type &gamma, const double denominatorA);
	//	-. for multiple independent observation sequences.
	/*virtual*/ void doEstimateObservationDensityParametersByML(const std::vector<size_t> &Ns, const unsigned int state, const std::vector<uivector_type> &observationSequences, const std::vector<dmatrix_type> &gammas, const size_t R, const double denominatorA);

	// MAP learning using conjugate prior.
	//	-. for a single independent observation sequence.
	/*virtual*/ void doEstimateObservationDensityParametersByMAPUsingConjugatePrior(const size_t N, const unsigned int state, const uivector_type &observations, const dmatrix_type &gamma, const double denominatorA);
	//	-. for multiple independent observation sequences.
	/*virtual*/ void doEstimateObservationDensityParametersByMAPUsingConjugatePrior(const std::vector<size_t> &Ns, const unsigned int state, const std::vector<uivector_type> &observationSequences, const std::vector<dmatrix_type> &gammas, const size_t R, const double denominatorA);

	// MAP learning using entropic prior.
	//	-. for a single independent observation sequence.
	/*virtual*/ void doEstimateObservationDensityParametersByMAPUsingEntropicPrior(const size_t N, const unsigned int state, const uivector_type &observations, const dmatrix_type &gamma, const double z, const bool doesTrimParameter, const double terminationTolerance, const size_t maxIteration, const double /*denominatorA*/);
	//	-. for multiple independent observation sequences.
	/*virtual*/ void doEstimateObservationDensityParametersByMAPUsingEntropicPrior(const std::vector<size_t> &Ns, const unsigned int state, const std::vector<uivector_type> &observationSequences, const std::vector<dmatrix_type> &gammas, const double z, const bool doesTrimParameter, const double terminationTolerance, const size_t maxIteration, const size_t R, const double /*denominatorA*/);

	//
	/*virtual*/ bool doReadObservationDensity(std::istream &stream);
	/*virtual*/ bool doWriteObservationDensity(std::ostream &stream) const;
	/*virtual*/ void doInitializeObservationDensity(const std::vector<double> &lowerBoundsOfObservationDensity, const std::vector<double> &upperBoundsOfObservationDensity);
	/*virtual*/ void doNormalizeObservationDensityParameters();

	/*virtual*/ bool doDoHyperparametersOfConjugatePriorExist() const
	{
		return base_type::doDoHyperparametersOfConjugatePriorExist() &&
			NULL != B_conj_.get();
	}

private:
	dmatrix_type B_;  // the observation(emission) probability distribution.

	// hyperparameters for the conjugate prior.
	//	[ref] "EM Algorithm 3 - THE EM Algorithm for MAP Estimates of HMM", personal note.
	//	[ref] "Pattern Recognition and Machine Learning", C. M. Bishop, Springer, 2006.
	boost::scoped_ptr<const dmatrix_type> B_conj_;  // beta.
};

}  // namespace swl


#endif  // __SWL_RND_UTIL__HMM_WITH_MULTINOMIAL_OBSERVATIONS__H_
