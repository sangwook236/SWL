#if !defined(__SWL_RND_UTIL__UNIVARIATE_NORMAL_MIXUTRE_MODEL__H_)
#define __SWL_RND_UTIL__UNIVARIATE_NORMAL_MIXUTRE_MODEL__H_ 1


#include "swl/rnd_util/ContinuousDensityMixtureModel.h"
#include <boost/random/linear_congruential.hpp>


namespace swl {

//--------------------------------------------------------------------------
// univariate normal mixture model

class SWL_RND_UTIL_API UnivariateNormalMixtureModel: public ContinuousDensityMixtureModel
{
public:
	typedef ContinuousDensityMixtureModel base_type;

public:
	UnivariateNormalMixtureModel(const size_t K);  // for ML learning.
	UnivariateNormalMixtureModel(const size_t K, const std::vector<double> &pi, const dvector_type &mus, const dvector_type &sigmas);
	UnivariateNormalMixtureModel(const size_t K, const std::vector<double> *pi_conj, const dvector_type *mus_conj, const dvector_type *betas_conj, const dvector_type *sigmas_conj, const dvector_type *nus_conj);  // for MAP learning using conjugate prior.
	virtual ~UnivariateNormalMixtureModel();

private:
	UnivariateNormalMixtureModel(const UnivariateNormalMixtureModel &rhs);  // not implemented.
	UnivariateNormalMixtureModel & operator=(const UnivariateNormalMixtureModel &rhs);  // not implemented.

public:
	//
	dvector_type & getMean()  {  return mus_;  }
	const dvector_type & getMean() const  {  return mus_;  }
	dvector_type & getStandardDeviation()  {  return  sigmas_;  }
	const dvector_type & getStandardDeviation() const  {  return  sigmas_;  }

protected:
	// if state == 0, hidden state = [ 1 0 0 ... 0 0 ].
	// if state == 1, hidden state = [ 0 1 0 ... 0 0 ].
	// ...
	// if state == N-1, hidden state = [ 0 0 0 ... 0 1 ].
	/*virtual*/ double doEvaluateMixtureComponentProbability(const unsigned int state, const dvector_type &observation) const;

	//
	/*virtual*/ void doGenerateObservationsSymbol(const unsigned int state, const size_t n, dmatrix_type &observations) const;
	// if seed != -1, the seed value is set.
	/*virtual*/ void doInitializeRandomSampleGeneration(const unsigned int seed = (unsigned int)-1) const;
	///*virtual*/ void doFinalizeRandomSampleGeneration() const;

	// ML learning.
	//	-. for IID observations.
	/*virtual*/ void doEstimateObservationDensityParametersByML(const size_t N, const unsigned int state, const dmatrix_type &observations, const dmatrix_type &gamma, const double sumGamma);

	// MAP learning using conjugate prior.
	//	-. for IID observations.
	/*virtual*/ void doEstimateObservationDensityParametersByMAPUsingConjugatePrior(const size_t N, const unsigned int state, const dmatrix_type &observations, const dmatrix_type &gamma, const double sumGamma);

	// MAP learning using entropic prior.
	//	-. for IID observations.
	/*virtual*/ void doEstimateObservationDensityParametersByMAPUsingEntropicPrior(const size_t N, const unsigned int state, const dmatrix_type &observations, const dmatrix_type &gamma, const double /*z*/, const bool /*doesTrimParameter*/, const bool isTrimmed, const double sumGamma);

	//
	/*virtual*/ bool doReadObservationDensity(std::istream &stream);
	/*virtual*/ bool doWriteObservationDensity(std::ostream &stream) const;
	/*virtual*/ void doInitializeObservationDensity(const std::vector<double> &lowerBoundsOfObservationDensity, const std::vector<double> &upperBoundsOfObservationDensity);
	/*virtual*/ void doNormalizeObservationDensityParameters()
	{
		// do nothing
	}

	/*virtual*/ bool doDoHyperparametersOfConjugatePriorExist() const
	{
		return base_type::doDoHyperparametersOfConjugatePriorExist() &&
			NULL != mus_conj_.get() && NULL != betas_conj_.get() && NULL != sigmas_conj_.get() && NULL != nus_conj_.get();
	}

protected:
	dvector_type mus_;  // the means of each components in the univariate normal mixture distribution.
	dvector_type sigmas_;  // the standard deviations of each components in the univariate normal mixture distribution.

	// hyperparameters for the conjugate prior.
	//	[ref] "Maximum a Posteriori Estimation for Multivariate Gaussian Mixture Observations of Markov Chains", J.-L. Gauvain adn C.-H. Lee, TSAP, 1994.
	//	[ref] "Pattern Recognition and Machine Learning", C. M. Bishop, Springer, 2006.
	//	[ref] "EM Algorithm 3 - THE EM Algorithm for MAP Estimates of HMM", personal note.
	boost::scoped_ptr<const dvector_type> mus_conj_;  // m.
	boost::scoped_ptr<const dvector_type> betas_conj_;  // beta. beta > 0.
	boost::scoped_ptr<const dvector_type> sigmas_conj_;  // inv(W).
	boost::scoped_ptr<const dvector_type> nus_conj_;  // nu. nu > D - 1.

	typedef boost::minstd_rand base_generator_type;
	mutable base_generator_type baseGenerator_;
};

}  // namespace swl


#endif  // __SWL_RND_UTIL__UNIVARIATE_NORMAL_MIXUTRE_MODEL__H_
