#if !defined(__SWL_RND_UTIL__DISCRETE_DENSITY_HMM__H_)
#define __SWL_RND_UTIL__DISCRETE_DENSITY_HMM__H_ 1


#include "swl/rnd_util/HMM.h"


namespace swl {

//--------------------------------------------------------------------------
// discrete density Hidden Markov Model (DDHMM).

class SWL_RND_UTIL_API DDHMM: public HMM
{
public:
	typedef HMM base_type;

protected:
	DDHMM(const size_t K, const size_t D);  // for ML learning.
	DDHMM(const size_t K, const size_t D, const dvector_type &pi, const dmatrix_type &A);
	DDHMM(const size_t K, const size_t D, const dvector_type *pi_conj, const dmatrix_type *A_conj);  // for MAP learning using conjugate prior.
public:
	virtual ~DDHMM();

private:
	DDHMM(const DDHMM &rhs);  // not implemented.
	DDHMM & operator=(const DDHMM &rhs);  // not implemented.

public:
	//
	static bool readSequence(std::istream &stream, size_t &N, uivector_type &observations);
	static bool writeSequence(std::ostream &stream, const uivector_type &observations);

	// forward algorithm without scaling.
	void runForwardAlgorithm(const size_t N, const uivector_type &observations, dmatrix_type &alpha, double &probability) const;
	// forward algorithm with scaling.
	// probability is the log probability.
	void runForwardAlgorithm(const size_t N, const uivector_type &observations, dvector_type &scale, dmatrix_type &alpha, double &probability) const;
	// backward algorithm without scaling.
	void runBackwardAlgorithm(const size_t N, const uivector_type &observations, dmatrix_type &beta, double &probability) const;
	// backward algorithm with scaling.
	// probability is the log probability.
	void runBackwardAlgorithm(const size_t N, const uivector_type &observations, const dvector_type &scale, dmatrix_type &beta, double &probability) const;

	// if useLog = true, probability is the log probability.
	void runViterbiAlgorithm(const size_t N, const uivector_type &observations, dmatrix_type &delta, uimatrix_type &psi, uivector_type &states, double &probability, const bool useLog = true) const;

	// ML learning.
	//	-. for a single independent observation sequence.
	bool trainByML(const size_t N, const uivector_type &observations, const double terminationTolerance, const size_t maxIteration, size_t &numIteration, double &initLogProbability, double &finalLogProbability);
	//	-. for multiple independent observation sequences.
	bool trainByML(const std::vector<size_t> &Ns, const std::vector<uivector_type> &observationSequences, const double terminationTolerance, const size_t maxIteration, size_t &numIteration, std::vector<double> &initLogProbabilities, std::vector<double> &finalLogProbabilities);

	// MAP learning using conjugate prior.
	//	-. for a single independent observation sequence.
	bool trainByMAPUsingConjugatePrior(const size_t N, const uivector_type &observations, const double terminationTolerance, const size_t maxIteration, size_t &numIteration, double &initLogProbability, double &finalLogProbability);
	//	-. for multiple independent observation sequences.
	bool trainByMAPUsingConjugatePrior(const std::vector<size_t> &Ns, const std::vector<uivector_type> &observationSequences, const double terminationTolerance, const size_t maxIteration, size_t &numIteration, std::vector<double> &initLogProbabilities, std::vector<double> &finalLogProbabilities);

	// MAP learning using entropic prior.
	//	-. for a single independent observation sequence.
	bool trainByMAPUsingEntropicPrior(const size_t N, const uivector_type &observations, const double z, const bool doesTrimParameter, const double terminationTolerance, const size_t maxIteration, size_t &numIteration, double &initLogProbability, double &finalLogProbability);
	//	-. for multiple independent observation sequences.
	bool trainByMAPUsingEntropicPrior(const std::vector<size_t> &Ns, const std::vector<uivector_type> &observationSequences, const double z, const bool doesTrimParameter, const double terminationTolerance, const size_t maxIteration, size_t &numIteration, std::vector<double> &initLogProbabilities, std::vector<double> &finalLogProbabilities);

	//
	void generateSample(const size_t N, uivector_type &observations, uivector_type &states, const unsigned int seed = (unsigned int)-1) const;

	//
	void computeXi(const size_t N, const uivector_type &observations, const dmatrix_type &alpha, const dmatrix_type &beta, std::vector<dmatrix_type> &xi) const;

protected:
	// if state == 0, hidden state = [ 1 0 0 ... 0 0 ].
	// if state == 1, hidden state = [ 0 1 0 ... 0 0 ].
	// ...
	// if state == N-1, hidden state = [ 0 0 0 ... 0 1 ].
	virtual double doEvaluateEmissionProbability(const unsigned int state, const unsigned int observation) const = 0;

	virtual unsigned int doGenerateObservationsSymbol(const unsigned int state) const = 0;

	// ML learning.
	//	-. for a single independent observation sequence.
	virtual void doEstimateObservationDensityParametersByML(const size_t N, const unsigned int state, const uivector_type &observations, const dmatrix_type &gamma, const double denominatorA) = 0;
	//	-. for multiple independent observation sequences.
	virtual void doEstimateObservationDensityParametersByML(const std::vector<size_t> &Ns, const unsigned int state, const std::vector<uivector_type> &observationSequences, const std::vector<dmatrix_type> &gammas, const size_t R, const double denominatorA) = 0;

	// MAP learning using conjugate prior.
	//	-. for a single independent observation sequence.
	virtual void doEstimateObservationDensityParametersByMAPUsingConjugatePrior(const size_t N, const unsigned int state, const uivector_type &observations, const dmatrix_type &gamma, const double denominatorA) = 0;
	//	-. for multiple independent observation sequences.
	virtual void doEstimateObservationDensityParametersByMAPUsingConjugatePrior(const std::vector<size_t> &Ns, const unsigned int state, const std::vector<uivector_type> &observationSequences, const std::vector<dmatrix_type> &gammas, const size_t R, const double denominatorA) = 0;

	// MAP learning using entropic prior.
	//	-. for a single independent observation sequence.
	virtual void doEstimateObservationDensityParametersByMAPUsingEntropicPrior(const size_t N, const unsigned int state, const uivector_type &observations, const dmatrix_type &gamma, const double z, const bool doesTrimParameter, const double terminationTolerance, const size_t maxIteration, const double denominatorA) = 0;
	//	-. for multiple independent observation sequences.
	virtual void doEstimateObservationDensityParametersByMAPUsingEntropicPrior(const std::vector<size_t> &Ns, const unsigned int state, const std::vector<uivector_type> &observationSequences, const std::vector<dmatrix_type> &gammas, const double z, const bool doesTrimParameter, const double terminationTolerance, const size_t maxIteration, const size_t R, const double denominatorA) = 0;

private:
	void runViterbiAlgorithmNotUsigLog(const size_t N, const uivector_type &observations, dmatrix_type &delta, uimatrix_type &psi, uivector_type &states, double &probability) const;
	void runViterbiAlgorithmUsingLog(const size_t N, const uivector_type &observations, dmatrix_type &delta, uimatrix_type &psi, uivector_type &states, double &probability) const;

protected:
};

}  // namespace swl


#endif  // __SWL_RND_UTIL__DISCRETE_DENSITY_HMM__H_
