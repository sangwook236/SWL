#if !defined(__SWL_RND_UTIL__HMM_WITH_UNIVARIATE_NORMAL_MIXTURE_OBSERVATIONS__H_)
#define __SWL_RND_UTIL__HMM_WITH_UNIVARIATE_NORMAL_MIXTURE_OBSERVATIONS__H_ 1


#include "swl/rnd_util/CDHMM.h"
#include "swl/rnd_util/HmmWithMixtureObservations.h"
#include <boost/random/linear_congruential.hpp>


namespace swl {

//--------------------------------------------------------------------------
// continuous density HMM with univariate normal mixture observation densities.

class SWL_RND_UTIL_API HmmWithUnivariateNormalMixtureObservations: public CDHMM, HmmWithMixtureObservations
{
public:
	typedef CDHMM base_type;
	//typedef HmmWithMixtureObservations base_type;
	typedef base_type::dmatrix_type dmatrix_type;

public:
	HmmWithUnivariateNormalMixtureObservations(const size_t K, const size_t C);  // for ML learning.
	HmmWithUnivariateNormalMixtureObservations(const size_t K, const size_t C, const dvector_type &pi, const dmatrix_type &A, const dmatrix_type &alphas, const dmatrix_type &mus, const dmatrix_type &sigmas);
	HmmWithUnivariateNormalMixtureObservations(const size_t K, const size_t C, const dvector_type *pi_conj, const dmatrix_type *A_conj, const dmatrix_type *alphas_conj, const dmatrix_type *mus_conj, const dmatrix_type *betas_conj, const dmatrix_type *sigmas_conj, const dmatrix_type *nus_conj);  // for MAP learning using conjugate prior.
	virtual ~HmmWithUnivariateNormalMixtureObservations();

private:
	HmmWithUnivariateNormalMixtureObservations(const HmmWithUnivariateNormalMixtureObservations &rhs);  // not implemented.
	HmmWithUnivariateNormalMixtureObservations & operator=(const HmmWithUnivariateNormalMixtureObservations &rhs);  // not implemented.

public:
	//
	dmatrix_type & getMean()  {  return mus_;  }
	const dmatrix_type & getMean() const  {  return mus_;  }
	dmatrix_type & getStandardDeviation()  {  return  sigmas_;  }
	const dmatrix_type & getStandardDeviation() const  {  return  sigmas_;  }

protected:
	// if state == 0, hidden state = [ 1 0 0 ... 0 0 ].
	// if state == 1, hidden state = [ 0 1 0 ... 0 0 ].
	// ...
	// if state == N-1, hidden state = [ 0 0 0 ... 0 1 ].
	/*virtual*/ double doEvaluateEmissionProbability(const unsigned int state, const boost::numeric::ublas::matrix_row<const dmatrix_type> &observation) const;
	// if seed != -1, the seed value is set.
	/*virtual*/ void doGenerateObservationsSymbol(const unsigned int state, boost::numeric::ublas::matrix_row<dmatrix_type> &observation, const unsigned int seed = (unsigned int)-1) const;

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
	/*virtual*/ void doEstimateObservationDensityParametersByMAPUsingEntropicPrior(const size_t N, const unsigned int state, const dmatrix_type &observations, const dmatrix_type &gamma, const double z, const double terminationTolerance, const size_t maxIteration, const double /*denominatorA*/);
	//	-. for multiple independent observation sequences.
	/*virtual*/ void doEstimateObservationDensityParametersByMAPUsingEntropicPrior(const std::vector<size_t> &Ns, const unsigned int state, const std::vector<dmatrix_type> &observationSequences, const std::vector<dmatrix_type> &gammas, const double z, const size_t R, const double terminationTolerance, const size_t maxIteration, const double /*denominatorA*/);

	//
	/*virtual*/ bool doReadObservationDensity(std::istream &stream);
	/*virtual*/ bool doWriteObservationDensity(std::ostream &stream) const;
	/*virtual*/ void doInitializeObservationDensity(const std::vector<double> &lowerBoundsOfObservationDensity, const std::vector<double> &upperBoundsOfObservationDensity);
	/*virtual*/ void doNormalizeObservationDensityParameters()
	{
		HmmWithMixtureObservations::normalizeObservationDensityParameters(K_);
	}

	/*virtual*/ bool doDoHyperparametersOfConjugatePriorExist() const
	{
		return base_type::doDoHyperparametersOfConjugatePriorExist() &&
			NULL != alphas_conj_.get() && NULL != mus_conj_.get() && NULL != betas_conj_.get() && NULL != sigmas_conj_.get() && NULL != nus_conj_.get();
	}

private:
	dmatrix_type mus_;  // the sets of means of each components in the univariate normal mixture distribution.
	dmatrix_type sigmas_;  // the sets of standard deviations of each components in the univariate normal mixture distribution.

	// hyperparameters for the conjugate prior.
	//	[ref] "EM Algorithm 3 - THE EM Algorithm for MAP Estimates of HMM", personal note.
	//	[ref] "Pattern Recognition and Machine Learning", C. M. Bishop, Springer, 2006.
	boost::scoped_ptr<const dmatrix_type> mus_conj_;  // m.
	boost::scoped_ptr<const dmatrix_type> betas_conj_;  // beta. beta > 0.
	boost::scoped_ptr<const dmatrix_type> sigmas_conj_;  // inv(W).
	boost::scoped_ptr<const dmatrix_type> nus_conj_;  // nu. nu > D - 1.

	typedef boost::minstd_rand base_generator_type;
	mutable base_generator_type baseGenerator_;
};

}  // namespace swl


#endif  // __SWL_RND_UTIL__HMM_WITH_UNIVARIATE_NORMAL_MIXTURE_OBSERVATIONS__H_
