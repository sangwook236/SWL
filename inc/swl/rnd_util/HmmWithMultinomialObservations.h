#if !defined(__SWL_RND_UTIL__HMM_WITH_MULTINOMIAL_OBSERVATIONS__H_)
#define __SWL_RND_UTIL__HMM_WITH_MULTINOMIAL_OBSERVATIONS__H_ 1


#include "swl/rnd_util/DDHMM.h"


namespace swl {

//--------------------------------------------------------------------------
// HMM with discrete multinomial observation densities

class SWL_RND_UTIL_API HmmWithMultinomialObservations: public DDHMM
{
public:
	typedef DDHMM base_type;

public:
	HmmWithMultinomialObservations(const size_t K, const size_t D);
	HmmWithMultinomialObservations(const size_t K, const size_t D, const std::vector<double> &pi, const boost::multi_array<double, 2> &A, const boost::multi_array<double, 2> &B);
	virtual ~HmmWithMultinomialObservations();

private:
	HmmWithMultinomialObservations(const HmmWithMultinomialObservations &rhs);
	HmmWithMultinomialObservations & operator=(const HmmWithMultinomialObservations &rhs);

public:
	/*virtual*/ bool estimateParameters(const size_t N, const std::vector<unsigned int> &observations, const double terminationTolerance, boost::multi_array<double, 2> &alpha, boost::multi_array<double, 2> &beta, boost::multi_array<double, 2> &gamma, size_t &numIteration, double &initLogProbability, double &finalLogProbability);

	//
	boost::multi_array<double, 2> & getObservationProbabilityMatrix()  {  return B_;  }
	const boost::multi_array<double, 2> & getObservationProbabilityMatrix() const  {  return B_;  }

protected:
	// if state == 0, hidden state = [ 1 0 0 ... 0 0 ]
	// if state == 1, hidden state = [ 0 1 0 ... 0 0 ]
	// ...
	// if state == N-1, hidden state = [ 0 0 0 ... 0 1 ]
	/*virtual*/ double evaluateEmissionProbability(const unsigned int state, const unsigned int observation) const
	{  return B_[state][observation];  }
	/*virtual*/ unsigned int generateObservationsSymbol(const unsigned int state) const;

	//
	/*virtual*/ bool readObservationDensity(std::istream &stream);
	/*virtual*/ bool writeObservationDensity(std::ostream &stream) const;
	/*virtual*/ void initializeObservationDensity();
	/*virtual*/ void normalizeObservationDensityParameters();

private:
	boost::multi_array<double, 2> B_;  // the observation(emission) probability distribution
};

}  // namespace swl


#endif  // __SWL_RND_UTIL__HMM_WITH_MULTINOMIAL_OBSERVATIONS__H_
