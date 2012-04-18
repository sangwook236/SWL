#if !defined(__SWL_RND_UTIL__HMM_WITH_UNIVARIATE_GAUSSIAN_OBSERVATIONS__H_)
#define __SWL_RND_UTIL__HMM_WITH_UNIVARIATE_GAUSSIAN_OBSERVATIONS__H_ 1


#include "swl/rnd_util/CDHMM.h"
#include <boost/random/linear_congruential.hpp>


namespace swl {

//--------------------------------------------------------------------------
// continuous density HMM with univariate Gaussian observation densities

class SWL_RND_UTIL_API HmmWithUnivariateGaussianObservations: public CDHMM
{
public:
	typedef CDHMM base_type;

public:
	HmmWithUnivariateGaussianObservations(const size_t K);
	HmmWithUnivariateGaussianObservations(const size_t K, const std::vector<double> &pi, const boost::multi_array<double, 2> &A, const std::vector<double> &mus, const std::vector<double> &sigmas);
	virtual ~HmmWithUnivariateGaussianObservations();

private:
	HmmWithUnivariateGaussianObservations(const HmmWithUnivariateGaussianObservations &rhs);
	HmmWithUnivariateGaussianObservations & operator=(const HmmWithUnivariateGaussianObservations &rhs);

public:
	// for a single independent observation sequence
	/*virtual*/ bool estimateParameters(const size_t N, const boost::multi_array<double, 2> &observations, const double terminationTolerance, boost::multi_array<double, 2> &alpha, boost::multi_array<double, 2> &beta, boost::multi_array<double, 2> &gamma, size_t &numIteration, double &initLogProbability, double &finalLogProbability);
	// for multiple independent observation sequences
	/*virtual*/ bool estimateParameters(const std::vector<size_t> &Ns, const std::vector<boost::multi_array<double, 2> > &observationSequences, const double terminationTolerance, size_t &numIteration,std::vector<double> &initLogProbabilities, std::vector<double> &finalLogProbabilities);

	//
	std::vector<double> & getMean()  {  return mus_;  }
	const std::vector<double> & getMean() const  {  return mus_;  }
	std::vector<double> & getStandardDeviation()  {  return  sigmas_;  }
	const std::vector<double> & getStandardDeviation() const  {  return  sigmas_;  }

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

private:
	std::vector<double> mus_;  // the means of the univariate Gaussian distribution
	std::vector<double> sigmas_;  // the standard deviations of the univariate Gaussian distribution

	typedef boost::minstd_rand base_generator_type;
	mutable base_generator_type baseGenerator_;
};

}  // namespace swl


#endif  // __SWL_RND_UTIL__HMM_WITH_UNIVARIATE_GAUSSIAN_OBSERVATIONS__H_
