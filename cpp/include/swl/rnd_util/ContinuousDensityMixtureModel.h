#if !defined(__SWL_RND_UTIL__CONTINUOUS_DENSITY_MXITURE_MODEL__H_)
#define __SWL_RND_UTIL__CONTINUOUS_DENSITY_MXITURE_MODEL__H_ 1


#include "swl/rnd_util/MixtureModel.h"
#include <boost/numeric/ublas/matrix.hpp>


namespace swl {

//--------------------------------------------------------------------------
// continuous density mixture model

class SWL_RND_UTIL_API ContinuousDensityMixtureModel: public MixtureModel
{
public:
	typedef MixtureModel base_type;
	typedef boost::numeric::ublas::vector<double> dvector_type;
	typedef boost::numeric::ublas::matrix<double> dmatrix_type;

protected:
	ContinuousDensityMixtureModel(const size_t K, const size_t D);  // for ML learning.
	ContinuousDensityMixtureModel(const size_t K, const size_t D, const std::vector<double> &pi);
	ContinuousDensityMixtureModel(const size_t K, const size_t D, const std::vector<double> *pi_conj);  // for MAP learning using conjugate prior.
public:
	virtual ~ContinuousDensityMixtureModel();

private:
	ContinuousDensityMixtureModel(const ContinuousDensityMixtureModel &rhs);  // not implemented.
	ContinuousDensityMixtureModel & operator=(const ContinuousDensityMixtureModel &rhs);  // not implemented.

public:
	//
	static bool readSequence(std::istream &stream, size_t &N, size_t &D, dmatrix_type &observations);
	static bool writeSequence(std::ostream &stream, const dmatrix_type &observations);

	// ML learning.
	//	-. for IID observations.
	bool trainByML(const size_t N, const dmatrix_type &observations, const double terminationTolerance, const size_t maxIteration, size_t &numIteration, double &initLogLikelihood, double &finalLogLikelihood);

	// MAP learning using conjugate prior.
	//	-. for IID observations.
	bool trainByMAPUsingConjugatePrior(const size_t N, const dmatrix_type &observations, const double terminationTolerance, const size_t maxIteration, size_t &numIteration, double &initLogLikelihood, double &finalLogLikelihood);

	// MAP learning using entropic prior.
	//	-. for IID observations.
	bool trainByMAPUsingEntropicPrior(const size_t N, const dmatrix_type &observations, const double z, const bool doesTrimParameter, const double terminationTolerance, const size_t maxIteration, size_t &numIteration, double &initLogLikelihood, double &finalLogLikelihood);

	// if seed != -1, the seed value is set
	void generateSample(const size_t N, dmatrix_type &observations, std::vector<unsigned int> &states, const unsigned int seed = (unsigned int)-1) const;

protected:
	// if state == 0, hidden state = [ 1 0 0 ... 0 0 ].
	// if state == 1, hidden state = [ 0 1 0 ... 0 0 ].
	// ...
	// if state == N-1, hidden state = [ 0 0 0 ... 0 1 ].
	double evaluateEmissionProbability(const dvector_type &observation) const;

	virtual double doEvaluateMixtureComponentProbability(const unsigned int state, const dvector_type &observation) const = 0;

	//
	virtual void doGenerateObservationsSymbol(const unsigned int state, const size_t n, dmatrix_type &observations) const = 0;

	// ML learning.
	//	-. for IID observations.
	virtual void doEstimateObservationDensityParametersByML(const size_t N, const unsigned int state, const dmatrix_type &observations, const dmatrix_type &gamma, const double sumGamma) = 0;

	// MAP learning using conjugate prior.
	//	-. for IID observations.
	virtual void doEstimateObservationDensityParametersByMAPUsingConjugatePrior(const size_t N, const unsigned int state, const dmatrix_type &observations, const dmatrix_type &gamma, const double sumGamma) = 0;

	// MAP learning using entropic prior.
	//	-. for IID observations.
	virtual void doEstimateObservationDensityParametersByMAPUsingEntropicPrior(const size_t N, const unsigned int state, const dmatrix_type &observations, const dmatrix_type &gamma, const double z, const bool doesTrimParameter, const bool isTrimmed, const double sumGamma) = 0;

private:
	void computeGamma(const std::size_t N, const dmatrix_type &observations, dmatrix_type &gamma, double &logLikelihood) const;
};

}  // namespace swl


#endif  // __SWL_RND_UTIL__CONTINUOUS_DENSITY_MXITURE_MODEL__H_
