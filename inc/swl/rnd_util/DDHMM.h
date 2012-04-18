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
	static bool readSequence(std::istream &stream, size_t &N, std::vector<unsigned int> &observations);
	static bool writeSequence(std::ostream &stream, const std::vector<unsigned int> &observations);

	// forward algorithm without scaling
	void runForwardAlgorithm(const size_t N, const std::vector<unsigned int> &observations, boost::multi_array<double, 2> &alpha, double &probability) const;
	// forward algorithm with scaling
	// probability is the LOG probability
	void runForwardAlgorithm(const size_t N, const std::vector<unsigned int> &observations, std::vector<double> &scale, boost::multi_array<double, 2> &alpha, double &probability) const;
	// backward algorithm without scaling
	void runBackwardAlgorithm(const size_t N, const std::vector<unsigned int> &observations, boost::multi_array<double, 2> &beta, double &probability) const;
	// backward algorithm with scaling
	// probability is the LOG probability
	void runBackwardAlgorithm(const size_t N, const std::vector<unsigned int> &observations, const std::vector<double> &scale, boost::multi_array<double, 2> &beta, double &probability) const;

	// if useLog = true, probability is the LOG probability
	void runViterbiAlgorithm(const size_t N, const std::vector<unsigned int> &observations, boost::multi_array<double, 2> &delta, boost::multi_array<unsigned int, 2> &psi, std::vector<unsigned int> &states, double &probability, const bool useLog = true) const;

	// for a single independent observation sequence
	bool estimateParameters(const size_t N, const std::vector<unsigned int> &observations, const double terminationTolerance, boost::multi_array<double, 2> &alpha, boost::multi_array<double, 2> &beta, boost::multi_array<double, 2> &gamma, size_t &numIteration, double &initLogProbability, double &finalLogProbability);
	// for multiple independent observation sequences
	bool estimateParameters(const std::vector<size_t> &Ns, const std::vector<std::vector<unsigned int> > &observationSequences, const double terminationTolerance, size_t &numIteration, std::vector<double> &initLogProbabilities, std::vector<double> &finalLogProbabilities);

	//
	void generateSample(const size_t N, std::vector<unsigned int> &observations, std::vector<unsigned int> &states) const;

	//
	void computeXi(const size_t N, const std::vector<unsigned int> &observations, const boost::multi_array<double, 2> &alpha, const boost::multi_array<double, 2> &beta, boost::multi_array<double, 3> &xi) const;

protected:
	// if state == 0, hidden state = [ 1 0 0 ... 0 0 ]
	// if state == 1, hidden state = [ 0 1 0 ... 0 0 ]
	// ...
	// if state == N-1, hidden state = [ 0 0 0 ... 0 1 ]
	virtual double doEvaluateEmissionProbability(const unsigned int state, const unsigned int observation) const = 0;
	virtual unsigned int doGenerateObservationsSymbol(const unsigned int state) const = 0;

	// for a single independent observation sequence
	virtual void doEstimateObservationDensityParametersInMStep(const size_t N, const std::vector<unsigned int> &observations, const boost::multi_array<double, 2> &gamma, const double denominatorA, const size_t k) = 0;
	// for multiple independent observation sequences
	virtual void doEstimateObservationDensityParametersInMStep(const std::vector<size_t> &Ns, const std::vector<std::vector<unsigned int> > &observationSequences, const std::vector<boost::multi_array<double, 2> > &gammas, const size_t R, const double denominatorA, const size_t k) = 0;

private:
	void runViterbiAlgorithmNotUsigLog(const size_t N, const std::vector<unsigned int> &observations, boost::multi_array<double, 2> &delta, boost::multi_array<unsigned int, 2> &psi, std::vector<unsigned int> &states, double &probability) const;
	void runViterbiAlgorithmUsingLog(const size_t N, const std::vector<unsigned int> &observations, boost::multi_array<double, 2> &delta, boost::multi_array<unsigned int, 2> &psi, std::vector<unsigned int> &states, double &probability) const;

protected:
};

}  // namespace swl


#endif  // __SWL_RND_UTIL__DISCRETE_DENSITY_HMM__H_
