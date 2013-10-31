#if !defined(__SWL_RND_UTIL__VON_MISES_MIXUTRE_MODEL__H_)
#define __SWL_RND_UTIL__VON_MISES_MIXUTRE_MODEL__H_ 1


#include "swl/rnd_util/ContinuousDensityMixtureModel.h"
#include <boost/smart_ptr.hpp>


namespace swl {
	
struct VonMisesTargetDistribution;
struct UnivariateNormalProposalDistribution;
struct UnivariateUniformProposalDistribution;

//--------------------------------------------------------------------------
// von Mises mixture model

class SWL_RND_UTIL_API VonMisesMixtureModel: public ContinuousDensityMixtureModel
{
public:
	typedef ContinuousDensityMixtureModel base_type;

public:
	VonMisesMixtureModel(const size_t K);  // for ML learning.
	VonMisesMixtureModel(const size_t K, const std::vector<double> &pi, const dvector_type &mus, const dvector_type &kappas);
	VonMisesMixtureModel(const size_t K, const std::vector<double> *pi_conj, const dvector_type *ms_conj, const dvector_type *Rs_conj, const dvector_type *cs_conj);  // for MAP learning using conjugate prior.
	virtual ~VonMisesMixtureModel();

private:
	VonMisesMixtureModel(const VonMisesMixtureModel &rhs);  // not implemented.
	VonMisesMixtureModel & operator=(const VonMisesMixtureModel &rhs);  // not implemented.

public:
	//
	dvector_type & getMeanDirection()  {  return mus_;  }
	const dvector_type & getMeanDirection() const  {  return mus_;  }
	dvector_type & getConcentrationParameter()  {  return  kappas_;  }
	const dvector_type & getConcentrationParameter() const  {  return  kappas_;  }

protected:
	// if state == 0, hidden state = [ 1 0 0 ... 0 0 ].
	// if state == 1, hidden state = [ 0 1 0 ... 0 0 ].
	// ...
	// if state == N-1, hidden state = [ 0 0 0 ... 0 1 ].
	/*virtual*/ double doEvaluateMixtureComponentProbability(const unsigned int state, const dvector_type &observation) const;

	//
	/*virtual*/ void doGenerateObservationsSymbol(const unsigned int state, boost::numeric::ublas::matrix_row<dmatrix_type> &observation) const;
	// if seed != -1, the seed value is set.
	/*virtual*/ void doInitializeRandomSampleGeneration(const unsigned int seed = (unsigned int)-1) const;
	/*virtual*/ void doFinalizeRandomSampleGeneration() const;

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
			NULL != ms_conj_.get() && NULL != Rs_conj_.get() && NULL != cs_conj_.get();
	}

protected:
	dvector_type mus_;  // the mean directions of the von Mises distribution. 0 <= mu < 2 * pi. [rad].
	dvector_type kappas_;  // the concentration parameters of the von Mises distribution. kappa >= 0.

	// hyperparameters for the conjugate prior.
	//	[ref] "Finding the Location of a Signal: A Bayesian Analysis", Peter Guttorp and Richard A. Lockhart, JASA, 1988.
	//	[ref] "EM Algorithm 3 - THE EM Algorithm for MAP Estimates of HMM", personal note.
	boost::scoped_ptr<const dvector_type> ms_conj_;  // m.
	boost::scoped_ptr<const dvector_type> Rs_conj_;  // R. R >= 0.
	boost::scoped_ptr<const dvector_type> cs_conj_;  // c. non-negative integer.

	mutable boost::scoped_ptr<VonMisesTargetDistribution> targetDist_;
#if 0
	mutable boost::scoped_ptr<UnivariateNormalProposalDistribution> proposalDist_;
#else
	mutable boost::scoped_ptr<UnivariateUniformProposalDistribution> proposalDist_;
#endif
};

}  // namespace swl


#endif  // __SWL_RND_UTIL__VON_MISES_MIXUTRE_MODEL__H_
