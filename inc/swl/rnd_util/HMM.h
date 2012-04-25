#if !defined(__SWL_RND_UTIL__HMM__H_)
#define __SWL_RND_UTIL__HMM__H_ 1


#include "swl/rnd_util/ExportRndUtil.h"
#include <boost/numeric/ublas/matrix.hpp>


namespace swl {

//--------------------------------------------------------------------------
// Hidden Markov Model (HMM)

class SWL_RND_UTIL_API HMM
{
public:
	//typedef HMM base_type;
	typedef boost::numeric::ublas::vector<double> dvector_type;
	typedef boost::numeric::ublas::matrix<double> dmatrix_type;
	typedef boost::numeric::ublas::vector<unsigned int> uivector_type;
	typedef boost::numeric::ublas::matrix<unsigned int> uimatrix_type;

protected:
	HMM(const size_t K, const size_t D);
	HMM(const size_t K, const size_t D, const dvector_type &pi, const dmatrix_type &A);
	virtual ~HMM();

private:
	HMM(const HMM &rhs);
	HMM & operator=(const HMM &rhs);

public:
	bool readModel(std::istream &stream);
	bool writeModel(std::ostream &stream) const;

	void initializeModel(const std::vector<double> &lowerBoundsOfObservationDensity, const std::vector<double> &upperBoundsOfObservationDensity);
	void normalizeModelParameters();

	void computeGamma(const size_t N, const dmatrix_type &alpha, const dmatrix_type &beta, dmatrix_type &gamma) const;

	//
	size_t getStateSize() const  {  return K_;  }
	size_t getObservationSize() const  {  return D_;  }

	dvector_type & getInitialStateDistribution()  {  return pi_;  }
	const dvector_type & getInitialStateDistribution() const  {  return pi_;  }
	dmatrix_type & getTransitionProbabilityMatrix()  {  return A_;  }
	const dmatrix_type & getTransitionProbabilityMatrix() const  {  return A_;  }

protected:
	virtual bool doReadObservationDensity(std::istream &stream) = 0;
	virtual bool doWriteObservationDensity(std::ostream &stream) const = 0;
	virtual void doInitializeObservationDensity(const std::vector<double> &lowerBoundsOfObservationDensity, const std::vector<double> &upperBoundsOfObservationDensity) = 0;
	virtual void doNormalizeObservationDensityParameters() = 0;

	unsigned int generateInitialState() const;
	unsigned int generateNextState(const unsigned int currState) const;

protected:
	const size_t K_;  // the dimension of hidden states
	const size_t D_;  // the dimension of observation symbols

	dvector_type pi_;  // the initial state distribution
	dmatrix_type A_;  // the state transition probability matrix
};

}  // namespace swl


#endif  // __SWL_RND_UTIL__HMM__H_
