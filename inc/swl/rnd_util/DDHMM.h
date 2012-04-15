#if !defined(__SWL_RND_UTIL__DISCRETE_DENSITY_HMM__H_)
#define __SWL_RND_UTIL__DISCRETE_DENSITY_HMM__H_ 1


#include "swl/rnd_util/HMM.h"


namespace swl {

//--------------------------------------------------------------------------
// discrete density Hidden Markov Model (DDHMM)

class SWL_RND_UTIL_API DDHMM: public HMM
{
public:
	typedef HMM base_type;

protected:
	DDHMM(const size_t K, const size_t D);
	DDHMM(const size_t K, const size_t D, const std::vector<double> &pi, const boost::multi_array<double, 2> &A);
public:
	virtual ~DDHMM();

private:
	DDHMM(const DDHMM &rhs);
	DDHMM & operator=(const DDHMM &rhs);

public:
	//
	static bool readSequence(std::istream &stream, size_t &N, std::vector<int> &observations);
	static bool writeSequence(std::ostream &stream, const std::vector<int> &observations);

	// forward algorithm without scaling
	void runForwardAlgorithm(const size_t N, const std::vector<int> &observations, boost::multi_array<double, 2> &alpha, double &probability) const;
	// forward algorithm with scaling
	// probability is the LOG probability
	void runForwardAlgorithm(const size_t N, const std::vector<int> &observations, std::vector<double> &scale, boost::multi_array<double, 2> &alpha, double &probability) const;
	// backward algorithm without scaling
	void runBackwardAlgorithm(const size_t N, const std::vector<int> &observations, boost::multi_array<double, 2> &beta, double &probability) const;
	// backward algorithm with scaling
	// probability is the LOG probability
	void runBackwardAlgorithm(const size_t N, const std::vector<int> &observations, const std::vector<double> &scale, boost::multi_array<double, 2> &beta, double &probability) const;

	// if useLog = true, probability is the LOG probability
	void runViterbiAlgorithm(const size_t N, const std::vector<int> &observations, boost::multi_array<double, 2> &delta, boost::multi_array<int, 2> &psi, std::vector<int> &states, double &probability, const bool useLog = true) const;

	//
	virtual bool estimateParameters(const size_t N, const std::vector<int> &observations, const double terminationTolerance, boost::multi_array<double, 2> &alpha, boost::multi_array<double, 2> &beta, boost::multi_array<double, 2> &gamma, size_t &numIteration, double &initLogProbability, double &finalLogProbability) = 0;

	//
	void generateSample(const size_t N, std::vector<int> &observations, std::vector<int> &states) const;

	//
	void computeXi(const size_t N, const std::vector<int> &observations, const boost::multi_array<double, 2> &alpha, const boost::multi_array<double, 2> &beta, boost::multi_array<double, 3> &xi) const;

protected:
	// if state == 0, hidden state = [ 1 0 0 ... 0 0 ]
	// if state == 1, hidden state = [ 0 1 0 ... 0 0 ]
	// ...
	// if state == N-1, hidden state = [ 0 0 0 ... 0 1 ]
	virtual double evaluateEmissionProbability(const int state, const int observation) const = 0;
	virtual int generateObservationsSymbol(const int state) const = 0;

private:
	void runViterbiAlgorithmNotUsigLog(const size_t N, const std::vector<int> &observations, boost::multi_array<double, 2> &delta, boost::multi_array<int, 2> &psi, std::vector<int> &states, double &probability) const;
	void runViterbiAlgorithmUsingLog(const size_t N, const std::vector<int> &observations, boost::multi_array<double, 2> &delta, boost::multi_array<int, 2> &psi, std::vector<int> &states, double &probability) const;

protected:
};

}  // namespace swl


#endif  // __SWL_RND_UTIL__DISCRETE_DENSITY_HMM__H_
