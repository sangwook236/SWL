#if !defined(__SWL_RND_UTIL__AR_HMM_WITH_MULTIVARIATE_NORMAL_OBSERVATIONS__H_)
#define __SWL_RND_UTIL__AR_HMM_WITH_MULTIVARIATE_NORMAL_OBSERVATIONS__H_ 1


#include "swl/rnd_util/CDHMM.h"
#include <gsl/gsl_rng.h>


namespace swl {

//--------------------------------------------------------------------------
// continuous density autoregressive (AR) HMM with multivariate normal observation densities.

class SWL_RND_UTIL_API ArHmmWithMultivariateNormalObservations: public CDHMM
{
public:
	typedef CDHMM base_type;

public:
	ArHmmWithMultivariateNormalObservations(const size_t K, const size_t D, const size_t P);  // for ML learning.
	ArHmmWithMultivariateNormalObservations(const size_t K, const size_t D, const size_t P, const dvector_type &pi, const dmatrix_type &A, const std::vector<dmatrix_type> &coeffs, const std::vector<dvector_type> &sigmas);
	// FIXME [fix] >> 
	ArHmmWithMultivariateNormalObservations(const size_t K, const size_t D, const size_t P, const dvector_type *pi_conj, const dmatrix_type *A_conj, const std::vector<dmatrix_type> *coeffs_conj);  // for MAP learning using conjugate prior.
	virtual ~ArHmmWithMultivariateNormalObservations();

private:
	ArHmmWithMultivariateNormalObservations(const ArHmmWithMultivariateNormalObservations &rhs);  // not implemented.
	ArHmmWithMultivariateNormalObservations & operator=(const ArHmmWithMultivariateNormalObservations &rhs);  // not implemented.

public:
	//
	size_t getAutoregressiveOrder() const { return P_; }

	std::vector<dmatrix_type> & getAutoregressiveCoefficient()  {  return coeffs_;  }
	const std::vector<dmatrix_type> & getAutoregressiveCoefficient() const  {  return coeffs_;  }
	std::vector<dvector_type> & getStandardDeviation()  {  return  sigmas_;  }
	const std::vector<dvector_type> & getStandardDeviation() const  {  return  sigmas_;  }

protected:
	// if state == 0, hidden state = [ 1 0 0 ... 0 0 ].
	// if state == 1, hidden state = [ 0 1 0 ... 0 0 ].
	// ...
	// if state == N-1, hidden state = [ 0 0 0 ... 0 1 ].
	/*virtual*/ double doEvaluateEmissionProbability(const unsigned int state, const dvector_type &observation) const;
	/*virtual*/ double doEvaluateEmissionProbability(const unsigned int state, const size_t n, const dmatrix_type &observations) const;

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
	/*virtual*/ void doEstimateObservationDensityParametersByMAPUsingEntropicPrior(const size_t N, const unsigned int state, const dmatrix_type &observations, const dmatrix_type &gamma, const double /*z*/, const bool /*doesTrimParameter*/, const double /*terminationTolerance*/, const size_t /*maxIteration*/, const double denominatorA);
	//	-. for multiple independent observation sequences.
	/*virtual*/ void doEstimateObservationDensityParametersByMAPUsingEntropicPrior(const std::vector<size_t> &Ns, const unsigned int state, const std::vector<dmatrix_type> &observationSequences, const std::vector<dmatrix_type> &gammas, const double /*z*/, const bool /*doesTrimParameter*/, const double /*terminationTolerance*/, const size_t /*maxIteration*/, const size_t R, const double denominatorA);

	//
	/*virtual*/ bool doReadObservationDensity(std::istream &stream);
	/*virtual*/ bool doWriteObservationDensity(std::ostream &stream) const;
	/*virtual*/ void doInitializeObservationDensity(const std::vector<double> &lowerBoundsOfObservationDensity, const std::vector<double> &upperBoundsOfObservationDensity);
	/*virtual*/ void doNormalizeObservationDensityParameters()
	{
		// do nothing.
	}

	/*virtual*/ bool doDoHyperparametersOfConjugatePriorExist() const
	{
		return base_type::doDoHyperparametersOfConjugatePriorExist() &&
			// FIXME [fix] >> 
			//NULL != mus_conj_.get() && NULL != betas_conj_.get() && NULL != sigmas_conj_.get() && NULL != nus_conj_.get();
			false;
	}

private:
	const size_t P_;  // the order of autoregressive model. p >= 1.

	std::vector<dmatrix_type> coeffs_;  // autoregression coefficients.
	std::vector<dvector_type> sigmas_;  // the variances of the input noise processes. sigam^2.

	// hyperparameters for the conjugate prior.
	//	[ref] "EM Algorithm 3 - THE EM Algorithm for MAP Estimates of HMM", personal note.
	//	[ref] "Pattern Recognition and Machine Learning", C. M. Bishop, Springer, 2006.
	// FIXME [implement] >> 
	boost::scoped_ptr<const std::vector<dmatrix_type> > coeffs_conj_;

	mutable gsl_rng *r_;
};

}  // namespace swl


#endif  // __SWL_RND_UTIL__AR_HMM_WITH_MULTIVARIATE_NORMAL_OBSERVATIONS__H_
