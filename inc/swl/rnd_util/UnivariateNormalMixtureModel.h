#if !defined(__SWL_RND_UTIL__UNIVARIATE_NORMAL_MIXUTRE_MODEL__H_)
#define __SWL_RND_UTIL__UNIVARIATE_NORMAL_MIXUTRE_MODEL__H_ 1


#include "swl/rnd_util/ContinuousDensityMixtureModel.h"
#include <boost/random/linear_congruential.hpp>


namespace swl {

//--------------------------------------------------------------------------
// normal mixture model

class SWL_RND_UTIL_API UnivariateNormalMixtureModel: public ContinuousDensityMixtureModel
{
public:
	typedef ContinuousDensityMixtureModel base_type;
	typedef base_type::dvector_type dvector_type;
	typedef base_type::dmatrix_type dmatrix_type;

public:
	UnivariateNormalMixtureModel(const size_t K);
	UnivariateNormalMixtureModel(const size_t K, const std::vector<double> &pi, const dvector_type &mus, const dvector_type &sigmas);
	virtual ~UnivariateNormalMixtureModel();

private:
	UnivariateNormalMixtureModel(const UnivariateNormalMixtureModel &rhs);  // not implemented
	UnivariateNormalMixtureModel & operator=(const UnivariateNormalMixtureModel &rhs);  // not implemented

public:
	//
	dvector_type & getMean()  {  return mus_;  }
	const dvector_type & getMean() const  {  return mus_;  }
	dvector_type & getStandardDeviation()  {  return  sigmas_;  }
	const dvector_type & getStandardDeviation() const  {  return  sigmas_;  }

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
	/*virtual*/ void doEstimateObservationDensityParametersByMAP(const size_t N, const unsigned int state, const dmatrix_type &observations, const dmatrix_type &gamma, const double sumGamma);

	//
	/*virtual*/ bool doReadObservationDensity(std::istream &stream);
	/*virtual*/ bool doWriteObservationDensity(std::ostream &stream) const;
	/*virtual*/ void doInitializeObservationDensity(const std::vector<double> &lowerBoundsOfObservationDensity, const std::vector<double> &upperBoundsOfObservationDensity);
	/*virtual*/ void doNormalizeObservationDensityParameters()
	{
		// do nothing
	}

protected:
	dvector_type mus_;  // the means of each components in the univariate normal mixture distribution.
	dvector_type sigmas_;  // the standard deviations of each components in the univariate normal mixture distribution.

	typedef boost::minstd_rand base_generator_type;
	mutable base_generator_type baseGenerator_;
};

}  // namespace swl


#endif  // __SWL_RND_UTIL__UNIVARIATE_NORMAL_MIXUTRE_MODEL__H_
