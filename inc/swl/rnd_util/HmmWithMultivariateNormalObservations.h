#if !defined(__SWL_RND_UTIL__HMM_WITH_MULTIVARIATE_NORMAL_OBSERVATIONS__H_)
#define __SWL_RND_UTIL__HMM_WITH_MULTIVARIATE_NORMAL_OBSERVATIONS__H_ 1


#include "swl/rnd_util/CDHMM.h"


namespace swl {

//--------------------------------------------------------------------------
// continuous density HMM with multivariate normal observation densities

class SWL_RND_UTIL_API HmmWithMultivariateNormalObservations: public CDHMM
{
public:
	typedef CDHMM base_type;

public:
	HmmWithMultivariateNormalObservations(const size_t K, const size_t D);
	HmmWithMultivariateNormalObservations(const size_t K, const size_t D, const std::vector<double> &pi, const boost::multi_array<double, 2> &A, const boost::multi_array<double, 2> &mus, const boost::multi_array<double, 3> &sigmas);
	virtual ~HmmWithMultivariateNormalObservations();

private:
	HmmWithMultivariateNormalObservations(const HmmWithMultivariateNormalObservations &rhs);
	HmmWithMultivariateNormalObservations & operator=(const HmmWithMultivariateNormalObservations &rhs);

public:
	//
	boost::multi_array<double, 2> & getMean()  {  return mus_;  }
	const boost::multi_array<double, 2> & getMean() const  {  return mus_;  }
	boost::multi_array<double, 3>& getCovarianceMatrix()  {  return  sigmas_;  }
	const boost::multi_array<double, 3> & getCovarianceMatrix() const  {  return  sigmas_;  }

protected:
	// if state == 0, hidden state = [ 1 0 0 ... 0 0 ]
	// if state == 1, hidden state = [ 0 1 0 ... 0 0 ]
	// ...
	// if state == N-1, hidden state = [ 0 0 0 ... 0 1 ]
	/*virtual*/ double doEvaluateEmissionProbability(const unsigned int state, const boost::multi_array<double, 2>::const_array_view<1>::type &observation) const;
	// if seed != -1, the seed value is set
	/*virtual*/ void doGenerateObservationsSymbol(const unsigned int state, boost::multi_array<double, 2>::array_view<1>::type &observation, const unsigned int seed = (unsigned int)-1) const;

	// for a single independent observation sequence
	/*virtual*/ void doEstimateObservationDensityParametersInMStep(const size_t N, const unsigned int state, const boost::multi_array<double, 2> &observations, boost::multi_array<double, 2> &gamma, const double denominatorA);
	// for multiple independent observation sequences
	/*virtual*/ void doEstimateObservationDensityParametersInMStep(const std::vector<size_t> &Ns, const unsigned int state, const std::vector<boost::multi_array<double, 2> > &observationSequences, const std::vector<boost::multi_array<double, 2> > &gammas, const size_t R, const double denominatorA);

	//
	/*virtual*/ bool doReadObservationDensity(std::istream &stream);
	/*virtual*/ bool doWriteObservationDensity(std::ostream &stream) const;
	/*virtual*/ void doInitializeObservationDensity();
	/*virtual*/ void doNormalizeObservationDensityParameters()
	{
		// do nothing
	}

private:
	boost::multi_array<double, 2> mus_;  // the mean vectors of each components in the multivariate normal mixture distribution
	boost::multi_array<double, 3> sigmas_;  // the covariance matrices of each components in the multivariate normal mixture distribution
};

}  // namespace swl


#endif  // __SWL_RND_UTIL__HMM_WITH_MULTIVARIATE_NORMAL_OBSERVATIONS__H_
