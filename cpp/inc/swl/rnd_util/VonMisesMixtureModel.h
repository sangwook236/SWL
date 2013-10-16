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
	typedef base_type::dvector_type dvector_type;
	typedef base_type::dmatrix_type dmatrix_type;

public:
	VonMisesMixtureModel(const size_t K);
	VonMisesMixtureModel(const size_t K, const std::vector<double> &pi, const dvector_type &mus, const dvector_type &kappas);
	virtual ~VonMisesMixtureModel();

private:
	VonMisesMixtureModel(const VonMisesMixtureModel &rhs);  // not implemented
	VonMisesMixtureModel & operator=(const VonMisesMixtureModel &rhs);  // not implemented

public:
	//
	dvector_type & getMeanDirection()  {  return mus_;  }
	const dvector_type & getMeanDirection() const  {  return mus_;  }
	dvector_type & getConcentrationParameter()  {  return  kappas_;  }
	const dvector_type & getConcentrationParameter() const  {  return  kappas_;  }

protected:
	// if state == 0, hidden state = [ 1 0 0 ... 0 0 ]
	// if state == 1, hidden state = [ 0 1 0 ... 0 0 ]
	// ...
	// if state == N-1, hidden state = [ 0 0 0 ... 0 1 ]
	/*virtual*/ double doEvaluateMixtureComponentProbability(const unsigned int state, const boost::numeric::ublas::matrix_row<const dmatrix_type> &observation) const;
	// if seed != -1, the seed value is set
	/*virtual*/ void doGenerateObservationsSymbol(const unsigned int state, boost::numeric::ublas::matrix_row<dmatrix_type> &observation, const unsigned int seed = (unsigned int)-1) const;

	// for IID observations
	/*virtual*/ void doEstimateObservationDensityParametersByML(const size_t N, const unsigned int state, const dmatrix_type &observations, const dmatrix_type &gamma, const double sumGamma);

	// for IID observations
	/*virtual*/ void doEstimateObservationDensityParametersByMAPUsingConjugatePrior(const size_t N, const unsigned int state, const dmatrix_type &observations, const dmatrix_type &gamma, const double sumGamma);

	//
	/*virtual*/ bool doReadObservationDensity(std::istream &stream);
	/*virtual*/ bool doWriteObservationDensity(std::ostream &stream) const;
	/*virtual*/ void doInitializeObservationDensity(const std::vector<double> &lowerBoundsOfObservationDensity, const std::vector<double> &upperBoundsOfObservationDensity);
	/*virtual*/ void doNormalizeObservationDensityParameters()
	{
		// do nothing
	}

protected:
	dvector_type mus_;  // the mean directions of the von Mises distribution. 0 <= mu < 2 * pi. [rad].
	dvector_type kappas_;  // the concentration parameters of the von Mises distribution. kappa >= 0.

	mutable boost::scoped_ptr<VonMisesTargetDistribution> targetDist_;
#if 0
	mutable boost::scoped_ptr<UnivariateNormalProposalDistribution> proposalDist_;
#else
	mutable boost::scoped_ptr<UnivariateUniformProposalDistribution> proposalDist_;
#endif
};

}  // namespace swl


#endif  // __SWL_RND_UTIL__VON_MISES_MIXUTRE_MODEL__H_
