#if !defined(__SWL_RND_UTIL__CONTINUOUS_DENSITY_HMM__H_)
#define __SWL_RND_UTIL__CONTINUOUS_DENSITY_HMM__H_ 1


#include "swl/rnd_util/HMM.h"


namespace swl {

//--------------------------------------------------------------------------
// continuous density Hidden Markov Model (CDHMM)

class SWL_RND_UTIL_API CDHMM: public HMM
{
public:
	typedef HMM base_type;

protected:
	CDHMM(const size_t K, const size_t D);
	CDHMM(const size_t K, const size_t D, const std::vector<double> &pi, const boost::multi_array<double, 2> &A);
public:
	virtual ~CDHMM();

private:
	CDHMM(const CDHMM &rhs);
	CDHMM & operator=(const CDHMM &rhs);

public:
	//
	static bool readSequence(std::istream &stream, size_t &N, size_t &D, boost::multi_array<double, 2> &observations);
	static bool writeSequence(std::ostream &stream, const boost::multi_array<double, 2> &observations);

	// forward algorithm without scaling
	void runForwardAlgorithm(const size_t N, const boost::multi_array<double, 2> &observations, boost::multi_array<double, 2> &alpha, double &probability) const;
	// forward algorithm with scaling
	// probability is the LOG probability
	void runForwardAlgorithm(const size_t N, const boost::multi_array<double, 2> &observations, std::vector<double> &scale, boost::multi_array<double, 2> &alpha, double &probability) const;
	// backward algorithm without scaling
	void runBackwardAlgorithm(const size_t N, const boost::multi_array<double, 2> &observations, boost::multi_array<double, 2> &beta, double &probability) const;
	// backward algorithm with scaling
	// probability is the LOG probability
	void runBackwardAlgorithm(const size_t N, const boost::multi_array<double, 2> &observations, const std::vector<double> &scale, boost::multi_array<double, 2> &beta, double &probability) const;

	// if useLog = true, probability is the LOG probability
	void runViterbiAlgorithm(const size_t N, const boost::multi_array<double, 2> &observations, boost::multi_array<double, 2> &delta, boost::multi_array<unsigned int, 2> &psi, std::vector<unsigned int> &states, double &probability, const bool useLog = true) const;

	// for a single independent observation sequence
	virtual bool estimateParameters(const size_t N, const boost::multi_array<double, 2> &observations, const double terminationTolerance, boost::multi_array<double, 2> &alpha, boost::multi_array<double, 2> &beta, boost::multi_array<double, 2> &gamma, size_t &numIteration, double &initLogProbability, double &finalLogProbability) = 0;
	// for multiple independent observation sequences
	virtual bool estimateParameters(const std::vector<size_t> &Ns, const std::vector<boost::multi_array<double, 2> > &observationSequences, const double terminationTolerance, size_t &numIteration, std::vector<double> &initLogProbabilities, std::vector<double> &finalLogProbabilities) = 0;

	// if seed != -1, the seed value is set
	void generateSample(const size_t N, boost::multi_array<double, 2> &observations, std::vector<unsigned int> &states, const unsigned int seed = (unsigned int)-1) const;

	//
	void computeXi(const size_t N, const boost::multi_array<double, 2> &observations, const boost::multi_array<double, 2> &alpha, const boost::multi_array<double, 2> &beta, boost::multi_array<double, 3> &xi) const;

protected:
	// if state == 0, hidden state = [ 1 0 0 ... 0 0 ]
	// if state == 1, hidden state = [ 0 1 0 ... 0 0 ]
	// ...
	// if state == N-1, hidden state = [ 0 0 0 ... 0 1 ]
	virtual double evaluateEmissionProbability(const unsigned int state, const boost::multi_array<double, 2>::const_array_view<1>::type &observation) const = 0;
	// if seed != -1, the seed value is set
	virtual void generateObservationsSymbol(const unsigned int state, boost::multi_array<double, 2>::array_view<1>::type &observation, const unsigned int seed = (unsigned int)-1) const = 0;

private:
	void runViterbiAlgorithmNotUsigLog(const size_t N, const boost::multi_array<double, 2> &observations, boost::multi_array<double, 2> &delta, boost::multi_array<unsigned int, 2> &psi, std::vector<unsigned int> &states, double &probability) const;
	void runViterbiAlgorithmUsingLog(const size_t N, const boost::multi_array<double, 2> &observations, boost::multi_array<double, 2> &delta, boost::multi_array<unsigned int, 2> &psi, std::vector<unsigned int> &states, double &probability) const;

protected:
};

}  // namespace swl


#endif  // __SWL_RND_UTIL__CONTINUOUS_DENSITY_HMM__H_
