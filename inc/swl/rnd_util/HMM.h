#if !defined(__SWL_RND_UTIL__HMM__H_)
#define __SWL_RND_UTIL__HMM__H_ 1


#include "swl/rnd_util/ExportRndUtil.h"
#include <boost/multi_array.hpp>


namespace swl {

//--------------------------------------------------------------------------
// Hidden Markov Model (HMM)

class SWL_RND_UTIL_API HMM
{
public:
	//typedef HMM base_type;

protected:
	HMM(const size_t K, const size_t D);
	HMM(const size_t K, const size_t D, const std::vector<double> &pi, const boost::multi_array<double, 2> &A);
	virtual ~HMM();

private:
	HMM(const HMM &rhs);
	HMM & operator=(const HMM &rhs);

public:
	bool readModel(std::istream &stream);
	bool writeModel(std::ostream &stream) const;

	void initializeModel();
	void normalizeModelParameters();

	void computeGamma(const size_t N, const boost::multi_array<double, 2> &alpha, const boost::multi_array<double, 2> &beta, boost::multi_array<double, 2> &gamma) const;

	//
	size_t getStateSize() const  {  return K_;  }
	size_t getObservationSize() const  {  return D_;  }

	std::vector<double> & getInitialStateDistribution()  {  return pi_;  }
	const std::vector<double> & getInitialStateDistribution() const  {  return pi_;  }
	boost::multi_array<double, 2> & getTransitionProbabilityMatrix()  {  return A_;  }
	const boost::multi_array<double, 2> & getTransitionProbabilityMatrix() const  {  return A_;  }

protected:
	virtual bool doReadObservationDensity(std::istream &stream) = 0;
	virtual bool doWriteObservationDensity(std::ostream &stream) const = 0;
	virtual void doInitializeObservationDensity() = 0;
	virtual void doNormalizeObservationDensityParameters() = 0;

	unsigned int generateInitialState() const;
	unsigned int generateNextState(const unsigned int currState) const;

protected:
	const size_t K_;  // the dimension of hidden states
	const size_t D_;  // the dimension of observation symbols

	std::vector<double> pi_;  // the initial state distribution
	boost::multi_array<double, 2> A_;  // the state transition probability matrix
};

}  // namespace swl


#endif  // __SWL_RND_UTIL__HMM__H_
