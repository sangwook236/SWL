#if !defined(__SWL_RND_UTIL__HMM_WITH_MULTIVARIATE_NORMAL_OBSERVATIONS__H_)
#define __SWL_RND_UTIL__HMM_WITH_MULTIVARIATE_NORMAL_OBSERVATIONS__H_ 1


#include "swl/rnd_util/CDHMM.h"
#include <gsl/gsl_rng.h>


namespace swl {

//--------------------------------------------------------------------------
// continuous density HMM with multivariate normal observation densities.

class SWL_RND_UTIL_API HmmWithMultivariateNormalObservations: public CDHMM
{
public:
	typedef CDHMM base_type;

public:
	HmmWithMultivariateNormalObservations(const size_t K, const size_t D);  // for ML learning.
	HmmWithMultivariateNormalObservations(const size_t K, const size_t D, const dvector_type &pi, const dmatrix_type &A, const std::vector<dvector_type> &mus, const std::vector<dmatrix_type> &sigmas);
	HmmWithMultivariateNormalObservations(const size_t K, const size_t D, const dvector_type *pi_conj, const dmatrix_type *A_conj, const std::vector<dvector_type> *mus_conj, const dvector_type *betas_conj, const std::vector<dmatrix_type> *sigmas_conj, const dvector_type *nus_conj);  // for MAP learning using conjugate prior.
	virtual ~HmmWithMultivariateNormalObservations();

private:
	HmmWithMultivariateNormalObservations(const HmmWithMultivariateNormalObservations &rhs);  // not implemented.
	HmmWithMultivariateNormalObservations & operator=(const HmmWithMultivariateNormalObservations &rhs);  // not implemented.

public:
	//
	std::vector<dvector_type> & getMean()  {  return mus_;  }
	const std::vector<dvector_type> & getMean() const  {  return mus_;  }
	std::vector<dmatrix_type> & getCovarianceMatrix()  {  return  sigmas_;  }
	const std::vector<dmatrix_type> & getCovarianceMatrix() const  {  return  sigmas_;  }

protected:
	// if state == 0, hidden state = [ 1 0 0 ... 0 0 ].
	// if state == 1, hidden state = [ 0 1 0 ... 0 0 ].
	// ...
	// if state == N-1, hidden state = [ 0 0 0 ... 0 1 ].
	/*virtual*/ double doEvaluateEmissionProbability(const unsigned int state, const dvector_type &observation) const;

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
			NULL != mus_conj_.get() && NULL != betas_conj_.get() && NULL != sigmas_conj_.get() && NULL != nus_conj_.get();
	}

private:
	std::vector<dvector_type> mus_;  // the mean vectors of each components in the multivariate normal mixture distribution.
	std::vector<dmatrix_type> sigmas_;  // the covariance matrices of each components in the multivariate normal mixture distribution.

	// hyperparameters for the conjugate prior.
	//	[ref] "EM Algorithm 3 - THE EM Algorithm for MAP Estimates of HMM", personal note.
	//	[ref] "Pattern Recognition and Machine Learning", C. M. Bishop, Springer, 2006.
	boost::scoped_ptr<const std::vector<dvector_type> > mus_conj_;  // m.
	boost::scoped_ptr<const dvector_type> betas_conj_;  // beta. beta > 0.
	boost::scoped_ptr<const std::vector<dmatrix_type> > sigmas_conj_;  // inv(W).
	boost::scoped_ptr<const dvector_type> nus_conj_;  // nu. nu > D - 1.

	mutable gsl_rng *r_;
};

}  // namespace swl


#endif  // __SWL_RND_UTIL__HMM_WITH_MULTIVARIATE_NORMAL_OBSERVATIONS__H_
