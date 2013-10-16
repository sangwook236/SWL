#if !defined(__SWL_RND_UTIL__HMM_WITH_MULTIVARIATE_NORMAL_MIXTURE_OBSERVATIONS__H_)
#define __SWL_RND_UTIL__HMM_WITH_MULTIVARIATE_NORMAL_MIXTURE_OBSERVATIONS__H_ 1


#include "swl/rnd_util/CDHMM.h"
#include "swl/rnd_util/HmmWithMixtureObservations.h"
#include <boost/multi_array.hpp>


namespace swl {

//--------------------------------------------------------------------------
// continuous density HMM with multivariate normal mixture observation densities.

class SWL_RND_UTIL_API HmmWithMultivariateNormalMixtureObservations: public CDHMM, HmmWithMixtureObservations
{
public:
	typedef CDHMM base_type;
	//typedef HmmWithMixtureObservations base_type;
	typedef base_type::dmatrix_type dmatrix_type;

public:
	HmmWithMultivariateNormalMixtureObservations(const size_t K, const size_t D, const size_t C);  // for ML learning.
	HmmWithMultivariateNormalMixtureObservations(const size_t K, const size_t D, const size_t C, const dvector_type &pi, const dmatrix_type &A, const dmatrix_type &alphas, const boost::multi_array<dvector_type, 2> &mus, const boost::multi_array<dmatrix_type, 2> &sigmas);
	HmmWithMultivariateNormalMixtureObservations(const size_t K, const size_t D, const size_t C, const dvector_type *pi_conj, const dmatrix_type *A_conj, const dmatrix_type *alphas_conj, const boost::multi_array<dvector_type, 2> *mus_conj, const dmatrix_type *betas_conj, const boost::multi_array<dmatrix_type, 2> *sigmas_conj, const dmatrix_type *nus_conj);  // for MAP learning using conjugate prior.
	virtual ~HmmWithMultivariateNormalMixtureObservations();

private:
	HmmWithMultivariateNormalMixtureObservations(const HmmWithMultivariateNormalMixtureObservations &rhs);  // not implemented.
	HmmWithMultivariateNormalMixtureObservations & operator=(const HmmWithMultivariateNormalMixtureObservations &rhs);  // not implemented.

public:
	//
	boost::multi_array<dvector_type, 2> & getMean()  {  return mus_;  }
	const boost::multi_array<dvector_type, 2> & getMean() const  {  return mus_;  }
	boost::multi_array<dmatrix_type, 2> & getCovarianceMatrix()  {  return  sigmas_;  }
	const boost::multi_array<dmatrix_type, 2> & getCovarianceMatrix() const  {  return  sigmas_;  }

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
	boost::multi_array<dvector_type, 2> mus_;  // the sets of mean vectors of each components in the multivariate normal mixture distribution.
	boost::multi_array<dmatrix_type, 2> sigmas_;  // the sets of covariance matrices of each components in the multivariate normal mixture distribution.

	// hyperparameters for the conjugate prior.
	//	[ref] "EM Algorithm 3 - THE EM Algorithm for MAP Estimates of HMM", personal note.
	//	[ref] "Pattern Recognition and Machine Learning", C. M. Bishop, Springer, 2006.
	boost::scoped_ptr<const boost::multi_array<dvector_type, 2> > mus_conj_;  // m.
	boost::scoped_ptr<const dmatrix_type> betas_conj_;  // beta. beta > 0.
	boost::scoped_ptr<const boost::multi_array<dmatrix_type, 2> > sigmas_conj_;  // inv(W).
	boost::scoped_ptr<const dmatrix_type> nus_conj_;  // nu. nu > D - 1.
};

}  // namespace swl


#endif  // __SWL_RND_UTIL__HMM_WITH_MULTIVARIATE_NORMAL_MIXTURE_OBSERVATIONS__H_
