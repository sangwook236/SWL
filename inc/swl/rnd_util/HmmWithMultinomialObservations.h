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
	typedef boost::numeric::ublas::vector<double> dvector_type;
	typedef boost::numeric::ublas::matrix<double> dmatrix_type;
	typedef boost::numeric::ublas::vector<unsigned int> uivector_type;
	typedef boost::numeric::ublas::matrix<unsigned int> uimatrix_type;

public:
	HmmWithMultinomialObservations(const size_t K, const size_t D);
	HmmWithMultinomialObservations(const size_t K, const size_t D, const dvector_type &pi, const dmatrix_type &A, const dmatrix_type &B);
	virtual ~HmmWithMultinomialObservations();

private:
	HmmWithMultinomialObservations(const HmmWithMultinomialObservations &rhs);
	HmmWithMultinomialObservations & operator=(const HmmWithMultinomialObservations &rhs);

public:
	//
	dmatrix_type & getObservationProbabilityMatrix()  {  return B_;  }
	const dmatrix_type & getObservationProbabilityMatrix() const  {  return B_;  }

protected:
	// if state == 0, hidden state = [ 1 0 0 ... 0 0 ]
	// if state == 1, hidden state = [ 0 1 0 ... 0 0 ]
	// ...
	// if state == N-1, hidden state = [ 0 0 0 ... 0 1 ]
	/*virtual*/ double doEvaluateEmissionProbability(const unsigned int state, const unsigned int observation) const
	{  return B_(state, observation);  }
	/*virtual*/ unsigned int doGenerateObservationsSymbol(const unsigned int state) const;

	// for a single independent observation sequence
	/*virtual*/ void doEstimateObservationDensityParametersInMStep(const size_t N, const unsigned int state, const uivector_type &observations, const dmatrix_type &gamma, const double denominatorA);
	// for multiple independent observation sequences
	/*virtual*/ void doEstimateObservationDensityParametersInMStep(const std::vector<size_t> &Ns, const unsigned int state, const std::vector<uivector_type> &observationSequences, const std::vector<dmatrix_type> &gammas, const size_t R, const double denominatorA);

	//
	/*virtual*/ bool doReadObservationDensity(std::istream &stream);
	/*virtual*/ bool doWriteObservationDensity(std::ostream &stream) const;
	/*virtual*/ void doInitializeObservationDensity(const std::vector<double> &lowerBoundsOfObservationDensity, const std::vector<double> &upperBoundsOfObservationDensity);
	/*virtual*/ void doNormalizeObservationDensityParameters();

private:
	dmatrix_type B_;  // the observation(emission) probability distribution
};

}  // namespace swl


#endif  // __SWL_RND_UTIL__HMM_WITH_MULTINOMIAL_OBSERVATIONS__H_
