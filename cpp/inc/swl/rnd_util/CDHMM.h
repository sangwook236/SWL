#if !defined(__SWL_RND_UTIL__CONTINUOUS_DENSITY_HMM__H_)
#define __SWL_RND_UTIL__CONTINUOUS_DENSITY_HMM__H_ 1


#include "swl/rnd_util/HMM.h"


namespace swl {

//--------------------------------------------------------------------------
// continuous density Hidden Markov Model (CDHMM).

class SWL_RND_UTIL_API CDHMM: public HMM
{
public:
	typedef HMM base_type;

protected:
	CDHMM(const size_t K, const size_t D);  // for ML learning.
	CDHMM(const size_t K, const size_t D, const dvector_type &pi, const dmatrix_type &A);
	CDHMM(const size_t K, const size_t D, const dvector_type *pi_conj, const dmatrix_type *A_conj);  // for MAP learning using conjugate prior.
public:
	virtual ~CDHMM();

private:
	CDHMM(const CDHMM &rhs);  // not implemented.
	CDHMM & operator=(const CDHMM &rhs);  // not implemented.

public:
	//
	static bool readSequence(std::istream &stream, size_t &N, size_t &D, dmatrix_type &observations);
	static bool writeSequence(std::ostream &stream, const dmatrix_type &observations);

	// forward algorithm without scaling.
	void runForwardAlgorithm(const size_t N, const dmatrix_type &observations, dmatrix_type &alpha, double &likelihood) const;
	// forward algorithm with scaling
	// probability is the log likelihood.
	void runForwardAlgorithm(const size_t N, const dmatrix_type &observations, dvector_type &scale, dmatrix_type &alpha, double &logLikelihood) const;
	// backward algorithm without scaling.
	void runBackwardAlgorithm(const size_t N, const dmatrix_type &observations, dmatrix_type &beta) const;
	// backward algorithm with scaling.
	void runBackwardAlgorithm(const size_t N, const dmatrix_type &observations, const dvector_type &scale, dmatrix_type &beta) const;

	// if useLog = true, probability is the log likelihood.
	void runViterbiAlgorithm(const size_t N, const dmatrix_type &observations, dmatrix_type &delta, uimatrix_type &psi, uivector_type &states, double &likelihood, const bool useLog = true) const;

	// ML learning.
	//	-. for a single independent observation sequence.
	bool trainByML(const size_t N, const dmatrix_type &observations, const double terminationTolerance, const size_t maxIteration, size_t &numIteration, double &initLogLikelihood, double &finalLogLikelihood);
	//	-. for multiple independent observation sequences.
	bool trainByML(const std::vector<size_t> &Ns, const std::vector<dmatrix_type> &observationSequences, const double terminationTolerance, const size_t maxIteration, size_t &numIteration, std::vector<double> &initLogLikelihoods, std::vector<double> &finalLogLikelihoods);

	// MAP learning using conjugate prior.
	//	-. for a single independent observation sequence.
	bool trainByMAPUsingConjugatePrior(const size_t N, const dmatrix_type &observations, const double terminationTolerance, const size_t maxIteration, size_t &numIteration, double &initLogLikelihood, double &finalLogLikelihood);
	//	-. for multiple independent observation sequences.
	bool trainByMAPUsingConjugatePrior(const std::vector<size_t> &Ns, const std::vector<dmatrix_type> &observationSequences, const double terminationTolerance, const size_t maxIteration, size_t &numIteration, std::vector<double> &initLogLikelihoods, std::vector<double> &finalLogLikelihoods);

	// MAP learning using entropic prior.
	//	-. for a single independent observation sequence.
	bool trainByMAPUsingEntropicPrior(const size_t N, const dmatrix_type &observations, const double z, const bool doesTrimParameter, const double terminationTolerance, const size_t maxIteration, size_t &numIteration, double &initLogLikelihood, double &finalLogLikelihood);
	//	-. for multiple independent observation sequences.
	bool trainByMAPUsingEntropicPrior(const std::vector<size_t> &Ns, const std::vector<dmatrix_type> &observationSequences, const double z, const bool doesTrimParameter, const double terminationTolerance, const size_t maxIteration, size_t &numIteration, std::vector<double> &initLogLikelihoods, std::vector<double> &finalLogLikelihoods);

	// if seed != -1, the seed value is set.
	void generateSample(const size_t N, dmatrix_type &observations, uivector_type &states, const unsigned int seed = (unsigned int)-1) const;

	//
	void computeXi(const size_t N, const dmatrix_type &observations, const dmatrix_type &alpha, const dmatrix_type &beta, std::vector<dmatrix_type> &xi) const;

protected:
	// if state == 0, hidden state = [ 1 0 0 ... 0 0 ].
	// if state == 1, hidden state = [ 0 1 0 ... 0 0 ].
	// ...
	// if state == N-1, hidden state = [ 0 0 0 ... 0 1 ].
	virtual double doEvaluateEmissionProbability(const unsigned int state, const dvector_type &observation) const = 0;

	//
	virtual void doGenerateObservationsSymbol(const unsigned int state, boost::numeric::ublas::matrix_row<dmatrix_type> &observation) const = 0;

	//
	virtual void doComputeObservationLikelihood(const size_t N, const dmatrix_type &observations, dmatrix_type &obsLikelihood) const;
	virtual void doComputeExpectedSufficientStatistics(const size_t N, const dmatrix_type &observations, const dmatrix_type &gamma, const std::vector<dmatrix_type> &xi, dvector_type &expNumVisits1, dvector_type &expNumVisitsN, dmatrix_type &expNumTrans/*, dmatrix_type &expNumEmit*/) const;
	virtual void doComputeExpectedSufficientStatistics(const std::vector<size_t> &Ns, const std::vector<dmatrix_type> &observationSequences, const std::vector<dmatrix_type> &gammas, const std::vector<std::vector<dmatrix_type> > &xis, dvector_type &expNumVisits1, dvector_type &expNumVisitsN, dmatrix_type &expNumTrans/*, dmatrix_type &expNumEmit*/) const;

	// ML learning.
	//	-. for a single independent observation sequence.
	virtual void doEstimateObservationDensityParametersByML(const size_t N, const unsigned int state, const dmatrix_type &observations, const dmatrix_type &gamma, const double denominatorA) = 0;
	//	-. for multiple independent observation sequences.
	virtual void doEstimateObservationDensityParametersByML(const std::vector<size_t> &Ns, const unsigned int state, const std::vector<dmatrix_type> &observationSequences, const std::vector<dmatrix_type> &gammas, const size_t R, const double denominatorA) = 0;

	// MAP learning using conjugate prior.
	//	-. for a single independent observation sequence.
	virtual void doEstimateObservationDensityParametersByMAPUsingConjugatePrior(const size_t N, const unsigned int state, const dmatrix_type &observations, const dmatrix_type &gamma, const double denominatorA) = 0;
	//	-. for multiple independent observation sequences.
	virtual void doEstimateObservationDensityParametersByMAPUsingConjugatePrior(const std::vector<size_t> &Ns, const unsigned int state, const std::vector<dmatrix_type> &observationSequences, const std::vector<dmatrix_type> &gammas, const size_t R, const double denominatorA) = 0;

	// MAP learning using entropic prior.
	//	-. for a single independent observation sequence.
	virtual void doEstimateObservationDensityParametersByMAPUsingEntropicPrior(const size_t N, const unsigned int state, const dmatrix_type &observations, const dmatrix_type &gamma, const double z, const bool doesTrimParameter, const double terminationTolerance, const size_t maxIteration, const double denominatorA) = 0;
	//	-. for multiple independent observation sequences.
	virtual void doEstimateObservationDensityParametersByMAPUsingEntropicPrior(const std::vector<size_t> &Ns, const unsigned int state, const std::vector<dmatrix_type> &observationSequences, const std::vector<dmatrix_type> &gammas, const double z, const bool doesTrimParameter, const double terminationTolerance, const size_t maxIteration, const size_t R, const double denominatorA) = 0;

private:
	void runViterbiAlgorithmNotUsigLog(const size_t N, const dmatrix_type &observations, dmatrix_type &delta, uimatrix_type &psi, uivector_type &states, double &probability) const;
	void runViterbiAlgorithmUsingLog(const size_t N, const dmatrix_type &observations, dmatrix_type &delta, uimatrix_type &psi, uivector_type &states, double &probability) const;

protected:
};

}  // namespace swl


#endif  // __SWL_RND_UTIL__CONTINUOUS_DENSITY_HMM__H_
