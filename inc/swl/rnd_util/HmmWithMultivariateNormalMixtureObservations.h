#if !defined(__SWL_RND_UTIL__HMM_WITH_MULTIVARIATE_NORMAL_MIXTURE_OBSERVATIONS__H_)
#define __SWL_RND_UTIL__HMM_WITH_MULTIVARIATE_NORMAL_MIXTURE_OBSERVATIONS__H_ 1


#include "swl/rnd_util/CDHMM.h"
#include "swl/rnd_util/HmmWithMixtureObservations.h"
#include <boost/multi_array.hpp>


namespace swl {

//--------------------------------------------------------------------------
// continuous density HMM with multivariate normal mixture observation densities

class SWL_RND_UTIL_API HmmWithMultivariateNormalMixtureObservations: public CDHMM, HmmWithMixtureObservations
{
public:
	typedef CDHMM base_type;
	//typedef HmmWithMixtureObservations base_type;
	typedef boost::numeric::ublas::vector<double> dvector_type;
	typedef boost::numeric::ublas::matrix<double> dmatrix_type;
	typedef boost::numeric::ublas::vector<unsigned int> uivector_type;
	typedef boost::numeric::ublas::matrix<unsigned int> uimatrix_type;

public:
	HmmWithMultivariateNormalMixtureObservations(const size_t K, const size_t D, const size_t C);
	HmmWithMultivariateNormalMixtureObservations(const size_t K, const size_t D, const size_t C, const dvector_type &pi, const dmatrix_type &A, const dmatrix_type &alphas, const boost::multi_array<dvector_type, 2> &mus, const boost::multi_array<dmatrix_type, 2> &sigmas);
	virtual ~HmmWithMultivariateNormalMixtureObservations();

private:
	HmmWithMultivariateNormalMixtureObservations(const HmmWithMultivariateNormalMixtureObservations &rhs);  // not implemented
	HmmWithMultivariateNormalMixtureObservations & operator=(const HmmWithMultivariateNormalMixtureObservations &rhs);  // not implemented

public:
	//
	boost::multi_array<dvector_type, 2> & getMean()  {  return mus_;  }
	const boost::multi_array<dvector_type, 2> & getMean() const  {  return mus_;  }
	boost::multi_array<dmatrix_type, 2> & getCovarianceMatrix()  {  return  sigmas_;  }
	const boost::multi_array<dmatrix_type, 2> & getCovarianceMatrix() const  {  return  sigmas_;  }

protected:
	// if state == 0, hidden state = [ 1 0 0 ... 0 0 ]
	// if state == 1, hidden state = [ 0 1 0 ... 0 0 ]
	// ...
	// if state == N-1, hidden state = [ 0 0 0 ... 0 1 ]
	/*virtual*/ double doEvaluateEmissionProbability(const unsigned int state, const boost::numeric::ublas::matrix_row<const dmatrix_type> &observation) const;
	// if seed != -1, the seed value is set
	/*virtual*/ void doGenerateObservationsSymbol(const unsigned int state, boost::numeric::ublas::matrix_row<dmatrix_type> &observation, const unsigned int seed = (unsigned int)-1) const;

	// for a single independent observation sequence
	/*virtual*/ void doEstimateObservationDensityParametersInMStep(const size_t N, const unsigned int state, const dmatrix_type &observations, dmatrix_type &gamma, const double denominatorA);
	// for multiple independent observation sequences
	/*virtual*/ void doEstimateObservationDensityParametersInMStep(const std::vector<size_t> &Ns, const unsigned int state, const std::vector<dmatrix_type> &observationSequences, const std::vector<dmatrix_type> &gammas, const size_t R, const double denominatorA);

	//
	/*virtual*/ bool doReadObservationDensity(std::istream &stream);
	/*virtual*/ bool doWriteObservationDensity(std::ostream &stream) const;
	/*virtual*/ void doInitializeObservationDensity(const std::vector<double> &lowerBoundsOfObservationDensity, const std::vector<double> &upperBoundsOfObservationDensity);
	/*virtual*/ void doNormalizeObservationDensityParameters()
	{
		HmmWithMixtureObservations::normalizeObservationDensityParameters(K_);
	}

private:
	boost::multi_array<dvector_type, 2> mus_;  // the sets of mean vectors of each components in the multivariate normal mixture distribution
	boost::multi_array<dmatrix_type, 2> sigmas_;  // the sets of covariance matrices of each components in the multivariate normal mixture distribution
};

}  // namespace swl


#endif  // __SWL_RND_UTIL__HMM_WITH_MULTIVARIATE_NORMAL_MIXTURE_OBSERVATIONS__H_
