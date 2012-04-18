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
	// for a single independent observation sequence
	/*virtual*/ bool estimateParameters(const size_t N, const boost::multi_array<double, 2> &observations, const double terminationTolerance, boost::multi_array<double, 2> &alpha, boost::multi_array<double, 2> &beta, boost::multi_array<double, 2> &gamma, size_t &numIteration, double &initLogProbability, double &finalLogProbability);
	// for multiple independent observation sequences
	/*virtual*/ bool estimateParameters(const std::vector<size_t> &Ns, const std::vector<boost::multi_array<double, 2> > &observationSequences, const double terminationTolerance, size_t &numIteration,std::vector<double> &initLogProbabilities, std::vector<double> &finalLogProbabilities);

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
	/*virtual*/ double evaluateEmissionProbability(const unsigned int state, const boost::multi_array<double, 2>::const_array_view<1>::type &observation) const;
	// if seed != -1, the seed value is set
	/*virtual*/ void generateObservationsSymbol(const unsigned int state, boost::multi_array<double, 2>::array_view<1>::type &observation, const unsigned int seed = (unsigned int)-1) const;

	//
	/*virtual*/ bool readObservationDensity(std::istream &stream);
	/*virtual*/ bool writeObservationDensity(std::ostream &stream) const;
	/*virtual*/ void initializeObservationDensity();
	/*virtual*/ void normalizeObservationDensityParameters()
	{
		// do nothing
	}

private:
	boost::multi_array<double, 2> mus_;  // the mean vectors of each components in the multivariate normal mixture distribution
	boost::multi_array<double, 3> sigmas_;  // the covariance matrices of each components in the multivariate normal mixture distribution
};

}  // namespace swl


#endif  // __SWL_RND_UTIL__HMM_WITH_MULTIVARIATE_NORMAL_OBSERVATIONS__H_
